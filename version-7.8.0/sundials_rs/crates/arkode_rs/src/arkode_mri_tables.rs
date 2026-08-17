//! Port of `src/arkode/arkode_mri_tables.c` (+ `src/arkode/arkode_mri_tables_impl.h`),
//! with `src/arkode/arkode_mri_tables.def` expanded at the bottom of the file
//! (the C build `#include`s the `.def` twice through the `ARK_MRI_TABLE`
//! X-macro; here each table body becomes one private `coeff_<NAME>` function
//! that both `MRIStepCoupling_LoadTable` and `MRIStepCoupling_LoadTableByName`
//! call).
//!
//! The `MRIStepCouplingMem` record, the `MRISTEP_METHOD_TYPE` enum and the
//! `ARKODE_MRITableID` identifiers are declared in
//! `include/arkode/arkode_mristep.h` but every routine that operates on them
//! lives in `arkode_mri_tables.c` — so they are folded in here, exactly the way
//! `ARKodeButcherTableMem` is folded into `arkode_butcher.rs` (contract §5).
//! `arkode_mristep.rs` consumes them with `use crate::arkode_mri_tables::*;`.
//!
//! Handle model: `MRIStepCoupling = Rc<RefCell<MRIStepCouplingMem>>`. The C
//! ragged arrays `sunrealtype*** W` / `sunrealtype*** G` become
//! `Vec<Vec<Vec<sunrealtype>>>` and `int** group` becomes `Vec<Vec<i32>>`;
//! an EMPTY outer `Vec` is C `NULL` (`if (MRIC->W)` -> `!MRIC.W.is_empty()`),
//! the same convention the Butcher table uses for its embedding row `d`.
//!
//! `MRIStepCoupling_Free` consumes the handle and drops it (C `free`s `c`, the
//! `W`/`G` matrices, `group`, then the struct); dropping the last `Rc` runs the
//! identical deallocation.

use std::cell::RefCell;
use std::rc::Rc;

use crate::arkode_butcher::{ARKodeButcherTable, ARKodeButcherTable_Alloc};
use crate::arkode_butcher_erk::{
    ARKodeButcherTable_LoadERK, ARKODE_EXPLICIT_MIDPOINT_EULER_2_1_2, ARKODE_FORWARD_EULER_1_1,
    ARKODE_HEUN_EULER_2_1_2, ARKODE_KNOTH_WOLKE_3_3, ARKODE_RALSTON_EULER_2_1_2,
};
use crate::arkode_impl::*;
use crate::arkode_mristep::{
    MRISTAGE_DIRK_FAST, MRISTAGE_DIRK_NOFAST, MRISTAGE_ERK_FAST, MRISTAGE_ERK_NOFAST,
    MRISTAGE_FIRST, MRISTAGE_STIFF_ACC,
};
use sundials_core::sundials_math::{SUNRabs, SUNRsqrt};
use sundials_core::sundials_types::*;
use sundials_core::sundials_utils::{sun_format_e, SUNFile};

/* ===========================================================================
 * MRIStep coupling types (include/arkode/arkode_mristep.h)
 * ===========================================================================*/

/// C `enum MRISTEP_METHOD_TYPE` — flag encoding the MRI method type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MRISTEP_METHOD_TYPE {
    #[default]
    MRISTEP_EXPLICIT,
    MRISTEP_IMPLICIT,
    MRISTEP_IMEX,
    MRISTEP_MERK,
    MRISTEP_SR,
}
pub use MRISTEP_METHOD_TYPE::*;

/// C `enum ARKODE_MRITableID` — MRI coupling table IDs.
///
/// Rendered as `i32` + constants rather than a Rust `enum` because the C enum
/// carries duplicate discriminants (`ARKODE_MIS_KW3` and `ARKODE_MIN_MRI_NUM`
/// are both 200, `ARKODE_MAX_MRI_NUM` aliases `ARKODE_IMEX_MRI_GARK_ARK2`),
/// which Rust `enum` variants cannot express.
pub type ARKODE_MRITableID = i32;

pub const ARKODE_MRI_NONE: ARKODE_MRITableID = -1; /* ensure enum is signed int */
pub const ARKODE_MIS_KW3: ARKODE_MRITableID = 200;
pub const ARKODE_MIN_MRI_NUM: ARKODE_MRITableID = 200;
pub const ARKODE_MRI_GARK_ERK33a: ARKODE_MRITableID = 201;
pub const ARKODE_MRI_GARK_ERK45a: ARKODE_MRITableID = 202;
pub const ARKODE_MRI_GARK_IRK21a: ARKODE_MRITableID = 203;
pub const ARKODE_MRI_GARK_ESDIRK34a: ARKODE_MRITableID = 204;
pub const ARKODE_MRI_GARK_ESDIRK46a: ARKODE_MRITableID = 205;
pub const ARKODE_IMEX_MRI_GARK3a: ARKODE_MRITableID = 206;
pub const ARKODE_IMEX_MRI_GARK3b: ARKODE_MRITableID = 207;
pub const ARKODE_IMEX_MRI_GARK4: ARKODE_MRITableID = 208;
pub const ARKODE_MRI_GARK_FORWARD_EULER: ARKODE_MRITableID = 209;
pub const ARKODE_MRI_GARK_RALSTON2: ARKODE_MRITableID = 210;
pub const ARKODE_MRI_GARK_ERK22a: ARKODE_MRITableID = 211;
pub const ARKODE_MRI_GARK_ERK22b: ARKODE_MRITableID = 212;
pub const ARKODE_MRI_GARK_RALSTON3: ARKODE_MRITableID = 213;
pub const ARKODE_MRI_GARK_BACKWARD_EULER: ARKODE_MRITableID = 214;
pub const ARKODE_MRI_GARK_IMPLICIT_MIDPOINT: ARKODE_MRITableID = 215;
pub const ARKODE_IMEX_MRI_GARK_EULER: ARKODE_MRITableID = 216;
pub const ARKODE_IMEX_MRI_GARK_TRAPEZOIDAL: ARKODE_MRITableID = 217;
pub const ARKODE_IMEX_MRI_GARK_MIDPOINT: ARKODE_MRITableID = 218;
pub const ARKODE_MERK21: ARKODE_MRITableID = 219;
pub const ARKODE_MERK32: ARKODE_MRITableID = 220;
pub const ARKODE_MERK43: ARKODE_MRITableID = 221;
pub const ARKODE_MERK54: ARKODE_MRITableID = 222;
pub const ARKODE_IMEX_MRI_SR21: ARKODE_MRITableID = 223;
pub const ARKODE_IMEX_MRI_SR32: ARKODE_MRITableID = 224;
pub const ARKODE_IMEX_MRI_SR43: ARKODE_MRITableID = 225;
pub const ARKODE_IMEX_MRI_GARK_ASCHER_ARK2: ARKODE_MRITableID = 226;
pub const ARKODE_IMEX_MRI_GARK_ARK2: ARKODE_MRITableID = 227;
pub const ARKODE_MAX_MRI_NUM: ARKODE_MRITableID = ARKODE_IMEX_MRI_GARK_ARK2;

/// C `struct MRIStepCouplingMem`.
///
/// `type` is a Rust keyword, so the field is `type_` (the workspace's
/// trailing-underscore rule for keyword collisions).
pub struct MRIStepCouplingMem {
    /// flag to encode the MRI method type
    pub type_: MRISTEP_METHOD_TYPE,
    /// number of MRI coupling matrices
    pub nmat: i32,
    /// size of coupling matrices ((stages+1) * stages)
    pub stages: i32,
    /// method order of accuracy
    pub q: i32,
    /// embedding order of accuracy
    pub p: i32,
    /// stage abscissae (EMPTY == C NULL)
    pub c: Vec<sunrealtype>,
    /// explicit coupling matrices \[nmat\]\[stages+1\]\[stages\] (EMPTY == C NULL)
    pub W: Vec<Vec<Vec<sunrealtype>>>,
    /// implicit coupling matrices \[nmat\]\[stages+1\]\[stages\] (EMPTY == C NULL)
    pub G: Vec<Vec<Vec<sunrealtype>>>,
    /// number of stage groups (MERK-specific)
    pub ngroup: i32,
    /// stages to integrate together (MERK-specific, EMPTY == C NULL)
    pub group: Vec<Vec<i32>>,
}

pub type MRIStepCoupling = Rc<RefCell<MRIStepCouplingMem>>;

/* ===========================================================================
 * Exported Functions
 * ===========================================================================*/

/*---------------------------------------------------------------
  Returns MRIStepCoupling table structure for pre-set MRI methods.

  Input:  imeth -- integer key for the desired method
  ---------------------------------------------------------------*/
pub fn MRIStepCoupling_LoadTable(method: ARKODE_MRITableID) -> Option<MRIStepCoupling> {
    match method {
        ARKODE_MRI_NONE => coeff_ARKODE_MRI_NONE(),
        ARKODE_MRI_GARK_FORWARD_EULER => coeff_ARKODE_MRI_GARK_FORWARD_EULER(),
        ARKODE_MRI_GARK_RALSTON2 => coeff_ARKODE_MRI_GARK_RALSTON2(),
        ARKODE_MIS_KW3 => coeff_ARKODE_MIS_KW3(),
        ARKODE_MRI_GARK_ERK22a => coeff_ARKODE_MRI_GARK_ERK22a(),
        ARKODE_MRI_GARK_ERK22b => coeff_ARKODE_MRI_GARK_ERK22b(),
        ARKODE_MRI_GARK_ERK33a => coeff_ARKODE_MRI_GARK_ERK33a(),
        ARKODE_MRI_GARK_RALSTON3 => coeff_ARKODE_MRI_GARK_RALSTON3(),
        ARKODE_MRI_GARK_ERK45a => coeff_ARKODE_MRI_GARK_ERK45a(),
        ARKODE_MRI_GARK_BACKWARD_EULER => coeff_ARKODE_MRI_GARK_BACKWARD_EULER(),
        ARKODE_MRI_GARK_IRK21a => coeff_ARKODE_MRI_GARK_IRK21a(),
        ARKODE_MRI_GARK_IMPLICIT_MIDPOINT => coeff_ARKODE_MRI_GARK_IMPLICIT_MIDPOINT(),
        ARKODE_MRI_GARK_ESDIRK34a => coeff_ARKODE_MRI_GARK_ESDIRK34a(),
        ARKODE_MRI_GARK_ESDIRK46a => coeff_ARKODE_MRI_GARK_ESDIRK46a(),
        ARKODE_IMEX_MRI_GARK_ASCHER_ARK2 => coeff_ARKODE_IMEX_MRI_GARK_ASCHER_ARK2(),
        ARKODE_IMEX_MRI_GARK_ARK2 => coeff_ARKODE_IMEX_MRI_GARK_ARK2(),
        ARKODE_IMEX_MRI_GARK_EULER => coeff_ARKODE_IMEX_MRI_GARK_EULER(),
        ARKODE_IMEX_MRI_GARK_TRAPEZOIDAL => coeff_ARKODE_IMEX_MRI_GARK_TRAPEZOIDAL(),
        ARKODE_IMEX_MRI_GARK_MIDPOINT => coeff_ARKODE_IMEX_MRI_GARK_MIDPOINT(),
        ARKODE_IMEX_MRI_GARK3a => coeff_ARKODE_IMEX_MRI_GARK3a(),
        ARKODE_IMEX_MRI_GARK3b => coeff_ARKODE_IMEX_MRI_GARK3b(),
        ARKODE_IMEX_MRI_GARK4 => coeff_ARKODE_IMEX_MRI_GARK4(),
        ARKODE_IMEX_MRI_SR21 => coeff_ARKODE_IMEX_MRI_SR21(),
        ARKODE_IMEX_MRI_SR32 => coeff_ARKODE_IMEX_MRI_SR32(),
        ARKODE_IMEX_MRI_SR43 => coeff_ARKODE_IMEX_MRI_SR43(),
        ARKODE_MERK21 => coeff_ARKODE_MERK21(),
        ARKODE_MERK32 => coeff_ARKODE_MERK32(),
        ARKODE_MERK43 => coeff_ARKODE_MERK43(),
        ARKODE_MERK54 => coeff_ARKODE_MERK54(),

        _ => {
            arkProcessError(
                None,
                ARK_ILL_INPUT,
                line!() as i32,
                "MRIStepCoupling_LoadTable",
                file!(),
                "Unknown coupling table",
            );
            None
        }
    }
}

/*---------------------------------------------------------------
  Returns MRIStepCoupling table structure for pre-set MRI methods.

  Input:  method -- string key for the desired method
  ---------------------------------------------------------------*/
pub fn MRIStepCoupling_LoadTableByName(method: &str) -> Option<MRIStepCoupling> {
    match method {
        "ARKODE_MRI_NONE" => return coeff_ARKODE_MRI_NONE(),
        "ARKODE_MRI_GARK_FORWARD_EULER" => return coeff_ARKODE_MRI_GARK_FORWARD_EULER(),
        "ARKODE_MRI_GARK_RALSTON2" => return coeff_ARKODE_MRI_GARK_RALSTON2(),
        "ARKODE_MIS_KW3" => return coeff_ARKODE_MIS_KW3(),
        "ARKODE_MRI_GARK_ERK22a" => return coeff_ARKODE_MRI_GARK_ERK22a(),
        "ARKODE_MRI_GARK_ERK22b" => return coeff_ARKODE_MRI_GARK_ERK22b(),
        "ARKODE_MRI_GARK_ERK33a" => return coeff_ARKODE_MRI_GARK_ERK33a(),
        "ARKODE_MRI_GARK_RALSTON3" => return coeff_ARKODE_MRI_GARK_RALSTON3(),
        "ARKODE_MRI_GARK_ERK45a" => return coeff_ARKODE_MRI_GARK_ERK45a(),
        "ARKODE_MRI_GARK_BACKWARD_EULER" => return coeff_ARKODE_MRI_GARK_BACKWARD_EULER(),
        "ARKODE_MRI_GARK_IRK21a" => return coeff_ARKODE_MRI_GARK_IRK21a(),
        "ARKODE_MRI_GARK_IMPLICIT_MIDPOINT" => return coeff_ARKODE_MRI_GARK_IMPLICIT_MIDPOINT(),
        "ARKODE_MRI_GARK_ESDIRK34a" => return coeff_ARKODE_MRI_GARK_ESDIRK34a(),
        "ARKODE_MRI_GARK_ESDIRK46a" => return coeff_ARKODE_MRI_GARK_ESDIRK46a(),
        "ARKODE_IMEX_MRI_GARK_ASCHER_ARK2" => return coeff_ARKODE_IMEX_MRI_GARK_ASCHER_ARK2(),
        "ARKODE_IMEX_MRI_GARK_ARK2" => return coeff_ARKODE_IMEX_MRI_GARK_ARK2(),
        "ARKODE_IMEX_MRI_GARK_EULER" => return coeff_ARKODE_IMEX_MRI_GARK_EULER(),
        "ARKODE_IMEX_MRI_GARK_TRAPEZOIDAL" => return coeff_ARKODE_IMEX_MRI_GARK_TRAPEZOIDAL(),
        "ARKODE_IMEX_MRI_GARK_MIDPOINT" => return coeff_ARKODE_IMEX_MRI_GARK_MIDPOINT(),
        "ARKODE_IMEX_MRI_GARK3a" => return coeff_ARKODE_IMEX_MRI_GARK3a(),
        "ARKODE_IMEX_MRI_GARK3b" => return coeff_ARKODE_IMEX_MRI_GARK3b(),
        "ARKODE_IMEX_MRI_GARK4" => return coeff_ARKODE_IMEX_MRI_GARK4(),
        "ARKODE_IMEX_MRI_SR21" => return coeff_ARKODE_IMEX_MRI_SR21(),
        "ARKODE_IMEX_MRI_SR32" => return coeff_ARKODE_IMEX_MRI_SR32(),
        "ARKODE_IMEX_MRI_SR43" => return coeff_ARKODE_IMEX_MRI_SR43(),
        "ARKODE_MERK21" => return coeff_ARKODE_MERK21(),
        "ARKODE_MERK32" => return coeff_ARKODE_MERK32(),
        "ARKODE_MERK43" => return coeff_ARKODE_MERK43(),
        "ARKODE_MERK54" => return coeff_ARKODE_MERK54(),
        _ => {}
    }

    arkProcessError(
        None,
        ARK_ILL_INPUT,
        line!() as i32,
        "MRIStepCoupling_LoadTableByName",
        file!(),
        "Unknown coupling table",
    );

    None
}

/*---------------------------------------------------------------
  Routine to allocate an empty MRIStepCoupling structure
  ---------------------------------------------------------------*/
pub fn MRIStepCoupling_Alloc(
    nmat: i32,
    stages: i32,
    type_: MRISTEP_METHOD_TYPE,
) -> Option<MRIStepCoupling> {
    let mut hasOmegas: sunbooleantype;
    let mut hasGammas: sunbooleantype;

    /* Check for legal input values */
    if nmat < 1 || stages < 1 {
        return None;
    }

    /* ------------------------------------------
     * Allocate and initialize coupling structure
     * ------------------------------------------ */

    let mut MRIC = MRIStepCouplingMem {
        type_,
        nmat,
        stages,
        q: 0,
        p: 0,
        c: Vec::new(),
        W: Vec::new(),
        G: Vec::new(),
        ngroup: 0,
        group: Vec::new(),
    };

    /* --------------------------------------------
     * Determine general storage format
     * -------------------------------------------- */

    hasOmegas = SUNFALSE;
    hasGammas = SUNFALSE;
    if (type_ == MRISTEP_EXPLICIT)
        || (type_ == MRISTEP_IMEX)
        || (type_ == MRISTEP_MERK)
        || (type_ == MRISTEP_SR)
    {
        hasOmegas = SUNTRUE;
    }
    if (type_ == MRISTEP_IMPLICIT) || (type_ == MRISTEP_IMEX) || (type_ == MRISTEP_SR) {
        hasGammas = SUNTRUE;
    }

    /* --------------------------------------------
     * Allocate abscissae and coupling coefficients
     * -------------------------------------------- */

    MRIC.c = vec![ZERO; stages as usize];

    if hasOmegas {
        /* allocate W matrices, their rows and their columns */
        MRIC.W = vec![vec![vec![ZERO; stages as usize]; stages as usize + 1]; nmat as usize];
    }

    if hasGammas {
        /* allocate G matrices, their rows and their columns */
        MRIC.G = vec![vec![vec![ZERO; stages as usize]; stages as usize + 1]; nmat as usize];
    }

    /* for MERK methods, allocate maximum possible number/sizes of stage groups */
    if type_ == MRISTEP_MERK {
        MRIC.ngroup = stages;
        MRIC.group = vec![vec![-1; stages as usize]; stages as usize];
    }

    Some(Rc::new(RefCell::new(MRIC)))
}

/*---------------------------------------------------------------
  Routine to allocate and fill an explicit, implicit, or ImEx
  MRIGARK MRIStepCoupling structure.
  ---------------------------------------------------------------*/
pub fn MRIStepCoupling_Create(
    nmat: i32,
    stages: i32,
    q: i32,
    p: i32,
    W_1d: &[sunrealtype],
    G_1d: &[sunrealtype],
    c_1d: &[sunrealtype],
) -> Option<MRIStepCoupling> {
    let type_: MRISTEP_METHOD_TYPE;

    /* Check for legal inputs (an EMPTY slice is C NULL) */
    if nmat < 1 || stages < 1 || c_1d.is_empty() {
        return None;
    }

    /* Check for method coefficients and set method type */
    if !W_1d.is_empty() && !G_1d.is_empty() {
        type_ = MRISTEP_IMEX;
    } else if !W_1d.is_empty() && G_1d.is_empty() {
        type_ = MRISTEP_EXPLICIT;
    } else if W_1d.is_empty() && !G_1d.is_empty() {
        type_ = MRISTEP_IMPLICIT;
    } else {
        return None;
    }

    /* Allocate MRIStepCoupling structure */
    let MRIC = MRIStepCoupling_Alloc(nmat, stages, type_)?;

    /* -------------------------
     * Copy the inputs into MRIC
     * ------------------------- */

    {
        let mut Cm = MRIC.borrow_mut();

        /* Method and embedding order */
        Cm.q = q;
        Cm.p = p;

        /* Abscissae */
        for i in 0..stages as usize {
            Cm.c[i] = c_1d[i];
        }

        /* Coupling coefficients stored as 1D arrays, based on whether they
           include embedding coefficients */
        if p == 0 {
            /* non-embedded method:  coupling coefficient 1D arrays have
               length nmat * stages * stages, with each stages * stages
               matrix stored in C (row-major) order */
            for k in 0..nmat as usize {
                for i in 0..stages as usize {
                    for j in 0..stages as usize {
                        if type_ == MRISTEP_EXPLICIT || type_ == MRISTEP_IMEX {
                            Cm.W[k][i][j] = W_1d[stages as usize * (stages as usize * k + i) + j];
                        }
                        if type_ == MRISTEP_IMPLICIT || type_ == MRISTEP_IMEX {
                            Cm.G[k][i][j] = G_1d[stages as usize * (stages as usize * k + i) + j];
                        }
                    }
                }
            }
        } else {
            /* embedded method:  coupling coefficient 1D arrays have
               length nmat * (stages+1) * stages, with each (stages+1) * stages
               matrix stored in C (row-major) order */
            for k in 0..nmat as usize {
                for i in 0..=stages as usize {
                    for j in 0..stages as usize {
                        if type_ == MRISTEP_EXPLICIT || type_ == MRISTEP_IMEX {
                            Cm.W[k][i][j] =
                                W_1d[(stages as usize + 1) * (stages as usize * k + i) + j];
                        }
                        if type_ == MRISTEP_IMPLICIT || type_ == MRISTEP_IMEX {
                            Cm.G[k][i][j] =
                                G_1d[(stages as usize + 1) * (stages as usize * k + i) + j];
                        }
                    }
                }
            }
        }
    }
    Some(MRIC)
}

/*---------------------------------------------------------------
  Construct the MRIGARK coupling matrix for an MIS method based
  on a given "slow" Butcher table.
  ---------------------------------------------------------------*/
pub fn MRIStepCoupling_MIStoMRI(
    B: Option<&ARKodeButcherTable>,
    q: i32,
    p: i32,
) -> Option<MRIStepCoupling> {
    let stages: i32;
    let mut padding: sunbooleantype;
    let mut Asum: sunrealtype;
    let mut type_: MRISTEP_METHOD_TYPE;

    let tol: sunrealtype = 100.0 * SUN_UNIT_ROUNDOFF;

    /* Check that input table is non-NULL */
    let B = B?;
    let Bm = B.borrow();

    /* If p>0, check that input table includes embedding coefficients */
    if (p > 0) && Bm.d.is_empty() {
        return None;
    }

    /* -----------------------------------
     * Check that the input table is valid
     * ----------------------------------- */

    /* First stage is just old solution */
    Asum = SUNRabs(Bm.c[0]);
    for j in 0..Bm.stages as usize {
        Asum += SUNRabs(Bm.A[0][j]);
    }
    if Asum > tol {
        return None;
    }

    /* Last stage exceeds 1 */
    if Bm.c[Bm.stages as usize - 1] > ONE + tol {
        return None;
    }

    /* All stages are sorted */
    for j in 1..Bm.stages as usize {
        if (Bm.c[j] - Bm.c[j - 1]) < -tol {
            return None;
        }
    }

    /* Each stage at most diagonally implicit */
    Asum = ZERO;
    for i in 0..Bm.stages as usize {
        for j in i + 1..Bm.stages as usize {
            Asum += SUNRabs(Bm.A[i][j]);
        }
    }
    if Asum > tol {
        return None;
    }

    /* -----------------------------------------
     * determine whether the table needs padding
     * ----------------------------------------- */

    padding = SUNFALSE;

    /* Pad if last stage does not equal 1 */
    if SUNRabs(Bm.c[Bm.stages as usize - 1] - ONE) > tol {
        padding = SUNTRUE;
    }

    /* Pad if last row of A does not equal b */
    for j in 0..Bm.stages as usize {
        if SUNRabs(Bm.A[Bm.stages as usize - 1][j] - Bm.b[j]) > tol {
            padding = SUNTRUE;
        }
    }

    /* If final stage is implicit and the method contains an embedding,
       we require padding since d != b */
    if (p > 0) && (SUNRabs(Bm.A[Bm.stages as usize - 1][Bm.stages as usize - 1]) > tol) {
        padding = SUNTRUE;
    }
    stages = if padding { Bm.stages + 1 } else { Bm.stages };

    /* -------------------------
     * determine the method type
     * ------------------------- */

    /* Check if the table is strictly lower triangular (explicit) */
    type_ = MRISTEP_EXPLICIT;

    for i in 0..Bm.stages as usize {
        for j in i..Bm.stages as usize {
            if SUNRabs(Bm.A[i][j]) > tol {
                type_ = MRISTEP_IMPLICIT;
            }
        }
    }

    /* ----------------------------
     * construct coupling structure
     * ---------------------------- */

    let MRIC = MRIStepCoupling_Alloc(1, stages, type_)?;

    {
        let mut Cm = MRIC.borrow_mut();

        /* Copy method/embedding orders */
        Cm.q = q;
        Cm.p = p;

        /* Copy abscissae, padding if needed */
        for i in 0..Bm.stages as usize {
            Cm.c[i] = Bm.c[i];
        }

        if padding {
            Cm.c[stages as usize - 1] = ONE;
        }

        /* Construct the coupling table (C aliases `C` with W or G) */
        let C: &mut Vec<Vec<sunrealtype>> = if type_ == MRISTEP_EXPLICIT {
            &mut Cm.W[0]
        } else {
            &mut Cm.G[0]
        };

        /* First row is identically zero */
        for i in 0..stages as usize {
            for j in 0..stages as usize {
                C[i][j] = ZERO;
            }
        }

        /* Remaining rows = A(2:end,:) - A(1:end-1,:) */
        for i in 1..Bm.stages as usize {
            for j in 0..Bm.stages as usize {
                C[i][j] = Bm.A[i][j] - Bm.A[i - 1][j];
            }
        }

        /* Padded row = b(:) - A(end,:) */
        if padding {
            for j in 0..Bm.stages as usize {
                C[stages as usize - 1][j] = Bm.b[j] - Bm.A[Bm.stages as usize - 1][j];
            }
        }

        /* Embedded row = d(:) - A(end,:) */
        if p > 0 {
            for j in 0..Bm.stages as usize {
                C[stages as usize][j] = Bm.d[j] - Bm.A[Bm.stages as usize - 1][j];
            }
        }
    }

    Some(MRIC)
}

/*---------------------------------------------------------------
  Routine to copy a MRIStepCoupling structure
  ---------------------------------------------------------------*/
pub fn MRIStepCoupling_Copy(MRIC: Option<&MRIStepCoupling>) -> Option<MRIStepCoupling> {
    let nmat: i32;
    let stages: i32;
    let type_: MRISTEP_METHOD_TYPE;

    /* Check for legal input */
    let MRIC = MRIC?;
    let Cm = MRIC.borrow();

    /* Copy method type */
    type_ = Cm.type_;

    /* Check for stage times */
    if Cm.c.is_empty() {
        return None;
    }

    /* Get the number of coupling matrices and stages */
    nmat = Cm.nmat;
    stages = Cm.stages;

    /* Allocate coupling structure */
    let MRICcopy = MRIStepCoupling_Alloc(nmat, stages, type_)?;

    {
        let mut cp = MRICcopy.borrow_mut();

        /* Copy method and embedding orders */
        cp.q = Cm.q;
        cp.p = Cm.p;

        /* Copy abscissae */
        for i in 0..stages as usize {
            cp.c[i] = Cm.c[i];
        }

        /* Copy explicit coupling matrices W */
        if !Cm.W.is_empty() {
            for k in 0..nmat as usize {
                for i in 0..=stages as usize {
                    for j in 0..stages as usize {
                        cp.W[k][i][j] = Cm.W[k][i][j];
                    }
                }
            }
        }

        /* Copy implicit coupling matrices G */
        if !Cm.G.is_empty() {
            for k in 0..nmat as usize {
                for i in 0..=stages as usize {
                    for j in 0..stages as usize {
                        cp.G[k][i][j] = Cm.G[k][i][j];
                    }
                }
            }
        }

        /* Copy MERK stage groups */
        if !Cm.group.is_empty() {
            cp.ngroup = Cm.ngroup;
            for i in 0..stages as usize {
                for j in 0..stages as usize {
                    cp.group[i][j] = Cm.group[i][j];
                }
            }
        }
    }

    Some(MRICcopy)
}

/*---------------------------------------------------------------
  Routine to query the MRIStepCoupling structure workspace size
  ---------------------------------------------------------------*/
pub fn MRIStepCoupling_Space(
    MRIC: Option<&MRIStepCoupling>,
    liw: &mut sunindextype,
    lrw: &mut sunindextype,
) {
    /* initialize outputs and return if MRIC is not allocated */
    *liw = 0;
    *lrw = 0;
    let MRIC = match MRIC {
        None => return,
        Some(MRIC) => MRIC,
    };
    let Cm = MRIC.borrow();

    /* fill outputs based on MRIC */
    *liw = 5;
    if !Cm.c.is_empty() {
        *lrw += Cm.stages as sunindextype;
    }
    if !Cm.W.is_empty() {
        *lrw += (Cm.nmat * (Cm.stages + 1) * Cm.stages) as sunindextype;
    }
    if !Cm.G.is_empty() {
        *lrw += (Cm.nmat * (Cm.stages + 1) * Cm.stages) as sunindextype;
    }
    if !Cm.group.is_empty() {
        *liw += (Cm.stages * Cm.stages) as sunindextype;
    }
}

/*---------------------------------------------------------------
  Routine to free a MRIStepCoupling structure
  ---------------------------------------------------------------*/
pub fn MRIStepCoupling_Free(MRIC: Option<MRIStepCoupling>) {
    /* Free each field within MRIStepCoupling structure, and then
       free structure itself -- dropping the handle does exactly this. */
    drop(MRIC);
}

/*---------------------------------------------------------------
  Routine to print a MRIStepCoupling structure
  ---------------------------------------------------------------*/
pub fn MRIStepCoupling_Write(MRIC: Option<&MRIStepCoupling>, outfile: &SUNFile) {
    /* check for valid coupling structure */
    let MRIC = match MRIC {
        None => return,
        Some(MRIC) => MRIC,
    };
    let Cm = MRIC.borrow();
    if Cm.W.is_empty() && Cm.G.is_empty() {
        return;
    }
    if Cm.c.is_empty() {
        return;
    }

    if !Cm.W.is_empty() {
        for i in 0..Cm.nmat as usize {
            if Cm.W[i].is_empty() {
                return;
            }
            for j in 0..=Cm.stages as usize {
                if Cm.W[i][j].is_empty() {
                    return;
                }
            }
        }
    }

    if !Cm.G.is_empty() {
        for i in 0..Cm.nmat as usize {
            if Cm.G[i].is_empty() {
                return;
            }
            for j in 0..=Cm.stages as usize {
                if Cm.G[i][j].is_empty() {
                    return;
                }
            }
        }
    }

    if !Cm.group.is_empty() {
        for i in 0..Cm.stages as usize {
            if Cm.group[i].is_empty() {
                return;
            }
        }
    }

    /* (the C `default: "  type = unknown\n"` arm is unreachable for a
       well-typed MRISTEP_METHOD_TYPE and is therefore omitted) */
    match Cm.type_ {
        MRISTEP_EXPLICIT => outfile.write_str("  type = explicit MRI\n"),
        MRISTEP_IMPLICIT => outfile.write_str("  type = implicit MRI\n"),
        MRISTEP_IMEX => outfile.write_str("  type = ImEx MRI\n"),
        MRISTEP_MERK => outfile.write_str("  type = MERK\n"),
        MRISTEP_SR => outfile.write_str("  type = MRISR\n"),
    }
    outfile.write_str(&format!("  nmat = {}\n", Cm.nmat));
    outfile.write_str(&format!("  stages = {}\n", Cm.stages));
    outfile.write_str(&format!("  method order (q) = {}\n", Cm.q));
    outfile.write_str(&format!("  embedding order (p) = {}\n", Cm.p));
    outfile.write_str("  c = ");
    for i in 0..Cm.stages as usize {
        outfile.write_str(&format!("{}  ", sun_format_e(Cm.c[i])));
    }
    outfile.write_str("\n");

    if !Cm.W.is_empty() {
        for k in 0..Cm.nmat as usize {
            outfile.write_str(&format!("  W[{}] = \n", k));
            for i in 0..=Cm.stages as usize {
                outfile.write_str("      ");
                for j in 0..Cm.stages as usize {
                    outfile.write_str(&format!("{}  ", sun_format_e(Cm.W[k][i][j])));
                }
                outfile.write_str("\n");
            }
            outfile.write_str("\n");
        }
    }

    if !Cm.G.is_empty() {
        for k in 0..Cm.nmat as usize {
            outfile.write_str(&format!("  G[{}] = \n", k));
            for i in 0..=Cm.stages as usize {
                outfile.write_str("      ");
                for j in 0..Cm.stages as usize {
                    outfile.write_str(&format!("{}  ", sun_format_e(Cm.G[k][i][j])));
                }
                outfile.write_str("\n");
            }
            outfile.write_str("\n");
        }
    }

    if !Cm.group.is_empty() {
        outfile.write_str(&format!("  ngroup = {}\n", Cm.ngroup));
        for i in 0..Cm.ngroup as usize {
            outfile.write_str(&format!("  group[{}] = ", i));
            for j in 0..Cm.stages as usize {
                if Cm.group[i][j] >= 0 {
                    outfile.write_str(&format!("{} ", Cm.group[i][j]));
                }
            }
            outfile.write_str("\n");
        }
    }
}

/* ===========================================================================
 * Private Functions
 * ===========================================================================*/

/* ---------------------------------------------------------------------------
 * Stage type identifier: returns one of the constants
 *
 * MRISTAGE_ERK_FAST    -- standard MIS-like stage
 * MRISTAGE_ERK_NOFAST  -- standard ERK stage
 * MRISTAGE_DIRK_NOFAST -- standard DIRK stage
 * MRISTAGE_DIRK_FAST   -- coupled DIRK + MIS-like stage
 * MRISTAGE_STIFF_ACC   -- "extra" stiffly-accurate stage
 *
 * for each nontrivial stage, or embedding stage, in an MRI-like method.
 * Otherwise (i.e., stage is not in [1,MRIC->stages]), returns
 * ARK_INVALID_TABLE (<0).
 *
 * The stage type is determined by 2 factors (for normal stages):
 * (a) Sum |MRIC->G[:][is][is]| (nonzero => DIRK)
 * (b) MRIC->c[is] - MRIC->c[is-1]  (nonzero => fast)
 * Similar tests are used for embedding stages.
 *
 * Note that MERK and MRI-SR methods do not use the stage-type identifiers,
 * so if those tables are input we just return MRISTAGE_ERK_FAST.
 * ---------------------------------------------------------------------------*/

pub fn mriStepCoupling_GetStageType(MRIC: &MRIStepCoupling, is: i32) -> i32 {
    let mut Gdiag: sunbooleantype = SUNFALSE;
    let mut Grow: sunbooleantype = SUNFALSE;
    let mut Wrow: sunbooleantype = SUNFALSE;
    let cdiff: sunbooleantype;
    let tol: sunrealtype = 100.0 * SUN_UNIT_ROUNDOFF;

    let Cm = MRIC.borrow();

    if (is < 0) || (is > Cm.stages) {
        return ARK_INVALID_TABLE;
    }

    if is == 0 {
        return MRISTAGE_FIRST;
    }

    /* report MRISTAGE_ERK_FAST for MERK and MRI-SR methods */
    if (Cm.type_ == MRISTEP_SR) || (Cm.type_ == MRISTEP_MERK) {
        return MRISTAGE_ERK_FAST;
    }

    let is_ = is as usize;

    /* separately handle an embedding "stage" from normal stages */
    if is < Cm.stages {
        /* normal */
        if !Cm.G.is_empty() {
            for i in 0..Cm.nmat as usize {
                Gdiag = Gdiag || (SUNRabs(Cm.G[i][is_][is_]) > tol);
                for j in 0..Cm.stages as usize {
                    Grow = Grow || (SUNRabs(Cm.G[i][is_][j]) > tol);
                }
            }
        }
        if !Cm.W.is_empty() {
            for i in 0..Cm.nmat as usize {
                for j in 0..Cm.stages as usize {
                    Wrow = Wrow || (SUNRabs(Cm.W[i][is_][j]) > tol);
                }
            }
        }

        /* abscissae difference */
        cdiff = SUNRabs(Cm.c[is_] - Cm.c[is_ - 1]) > tol;
    } else {
        /* embedding */
        if !Cm.G.is_empty() {
            for i in 0..Cm.nmat as usize {
                Gdiag = Gdiag || (SUNRabs(Cm.G[i][is_][is_ - 1]) > tol);
                for j in 0..Cm.stages as usize {
                    Grow = Grow || (SUNRabs(Cm.G[i][is_][j]) > tol);
                }
            }
        }
        if !Cm.W.is_empty() {
            for i in 0..Cm.nmat as usize {
                for j in 0..Cm.stages as usize {
                    Wrow = Wrow || (SUNRabs(Cm.W[i][is_][j]) > tol);
                }
            }
        }
        cdiff = SUNRabs(Cm.c[is_ - 1] - Cm.c[is_ - 2]) > tol;
    }

    /* make determination */
    if !(Gdiag || Grow || Wrow || cdiff) && (is > 0) {
        /* stiffly-accurate stage */
        return MRISTAGE_STIFF_ACC;
    }
    if Gdiag {
        /* DIRK */
        if cdiff {
            /* Fast */
            MRISTAGE_DIRK_FAST
        } else {
            MRISTAGE_DIRK_NOFAST
        }
    } else {
        /* ERK */
        if cdiff {
            /* Fast */
            MRISTAGE_ERK_FAST
        } else {
            MRISTAGE_ERK_NOFAST
        }
    }
}

/* ---------------------------------------------------------------------------
 * Computes the stage RHS vector storage maps. With repeated abscissae the
 * first stage of the pair generally corresponds to a column of zeros and so
 * does not need to be computed and stored. The stage_map indicates if the RHS
 * needs to be computed and where to store it i.e., stage_map[i] > -1.
 *
 * Note: for MERK and MRI-SR methods, this should be an "identity" map, and all
 * stage vectors should be allocated.
 * ---------------------------------------------------------------------------*/

pub fn mriStepCoupling_GetStageMap(
    MRIC: Option<&MRIStepCoupling>,
    stage_map: &mut [i32],
    nstages_active: &mut i32,
) -> i32 {
    let mut idx: i32;
    let mut Wsum: sunrealtype;
    let mut Gsum: sunrealtype;
    let tol: sunrealtype = 100.0 * SUN_UNIT_ROUNDOFF;

    /* ----------------------
     * Check for valid inputs
     * ---------------------- */

    /* (the C `!stage_map || !nstages_active` NULL tests are unrepresentable
       for `&mut` arguments and are therefore dropped) */
    let MRIC = match MRIC {
        None => return ARK_ILL_INPUT,
        Some(MRIC) => MRIC,
    };
    let Cm = MRIC.borrow();
    if Cm.W.is_empty() && Cm.G.is_empty() {
        return ARK_ILL_INPUT;
    }

    /* -------------------------------------------
     * MERK and MRI-SR have "identity" storage map
     * ------------------------------------------- */

    if (Cm.type_ == MRISTEP_MERK) || (Cm.type_ == MRISTEP_SR) {
        /* Number of stage RHS vectors active */
        *nstages_active = Cm.stages;

        /* Create an identity map (all columns are non-zero) */
        for j in 0..Cm.stages as usize {
            stage_map[j] = j as i32;
        }
        return ARK_SUCCESS;
    }

    /* ----------------------------------------
     * Compute storage map for MRI-GARK methods
     * ---------------------------------------- */

    /* Number of stage RHS vectors active */
    *nstages_active = 0;

    /* Initial storage index */
    idx = 0;

    /* Check if a stage corresponds to a column of zeros for all coupling
     * matrices by computing the column sums */
    for j in 0..Cm.stages as usize {
        Wsum = ZERO;
        Gsum = ZERO;

        if !Cm.W.is_empty() {
            for k in 0..Cm.nmat as usize {
                for i in 0..=Cm.stages as usize {
                    Wsum += SUNRabs(Cm.W[k][i][j]);
                }
            }
        }

        if !Cm.G.is_empty() {
            for k in 0..Cm.nmat as usize {
                for i in 0..=Cm.stages as usize {
                    Gsum += SUNRabs(Cm.G[k][i][j]);
                }
            }
        }

        if Wsum > tol || Gsum > tol {
            stage_map[j] = idx;
            idx += 1;
        } else {
            stage_map[j] = -1;
        }
    }

    /* Check and set number of stage RHS vectors active */
    if idx < 1 {
        return ARK_ILL_INPUT;
    }

    *nstages_active = idx;

    ARK_SUCCESS
}

/* ===========================================================================
 * Built-in coupling tables (src/arkode/arkode_mri_tables.def)
 *
 * One function per `ARK_MRI_TABLE(name, coeff)` entry, in the order the C
 * `.def` file declares them. Every coefficient is transcribed digit-for-digit
 * with the original arithmetic order preserved (`SUN_RCONST(x)` is the identity
 * for `sunrealtype = double`).
 * ===========================================================================*/

fn coeff_ARKODE_MRI_NONE() -> Option<MRIStepCoupling> {
    None
}

fn coeff_ARKODE_MRI_GARK_FORWARD_EULER() -> Option<MRIStepCoupling> {
    let B = ARKodeButcherTable_LoadERK(ARKODE_FORWARD_EULER_1_1)?;
    let (q, p) = {
        let Bm = B.borrow();
        (Bm.q, Bm.p)
    };
    let C = MRIStepCoupling_MIStoMRI(Some(&B), q, p);
    drop(B); /* ARKodeButcherTable_Free(B) */
    C
}

fn coeff_ARKODE_MRI_GARK_RALSTON2() -> Option<MRIStepCoupling> {
    /* Roberts et al., SISC 44:A1405 - A1427, 2022 */
    let B = ARKodeButcherTable_LoadERK(ARKODE_RALSTON_EULER_2_1_2)?;
    let (q, p) = {
        let Bm = B.borrow();
        (Bm.q, Bm.p)
    };
    let C = MRIStepCoupling_MIStoMRI(Some(&B), q, p);
    drop(B); /* ARKodeButcherTable_Free(B) */
    C
}

fn coeff_ARKODE_MIS_KW3() -> Option<MRIStepCoupling> {
    /* Schlegel et al., JCAM 226:345-357, 2009 */
    let B = ARKodeButcherTable_LoadERK(ARKODE_KNOTH_WOLKE_3_3)?;
    let (q, p) = {
        let Bm = B.borrow();
        (Bm.q, Bm.p)
    };
    let C = MRIStepCoupling_MIStoMRI(Some(&B), q, p);
    drop(B); /* ARKodeButcherTable_Free(B) */
    C
}

fn coeff_ARKODE_MRI_GARK_ERK22a() -> Option<MRIStepCoupling> {
    /* A. Sandu, SINUM 57:2300-2327, 2019 */
    let B = ARKodeButcherTable_LoadERK(ARKODE_EXPLICIT_MIDPOINT_EULER_2_1_2)?;
    let (q, p) = {
        let Bm = B.borrow();
        (Bm.q, Bm.p)
    };
    let C = MRIStepCoupling_MIStoMRI(Some(&B), q, p);
    drop(B); /* ARKodeButcherTable_Free(B) */
    C
}

fn coeff_ARKODE_MRI_GARK_ERK22b() -> Option<MRIStepCoupling> {
    /* A. Sandu, SINUM 57:2300-2327, 2019 */
    let B = ARKodeButcherTable_LoadERK(ARKODE_HEUN_EULER_2_1_2)?;
    let (q, p) = {
        let Bm = B.borrow();
        (Bm.q, Bm.p)
    };
    let C = MRIStepCoupling_MIStoMRI(Some(&B), q, p);
    drop(B); /* ARKodeButcherTable_Free(B) */
    C
}

fn coeff_ARKODE_MRI_GARK_ERK33a() -> Option<MRIStepCoupling> {
    /* A. Sandu, SINUM 57:2300-2327, 2019 */
    let C = MRIStepCoupling_Alloc(2, 4, MRISTEP_EXPLICIT)?;
    {
        let mut Cm = C.borrow_mut();

        Cm.q = 3;
        Cm.p = 2;

        Cm.c[1] = ONE / 3.0;
        Cm.c[2] = TWO / 3.0;
        Cm.c[3] = ONE;

        Cm.W[0][1][0] = ONE / 3.0;
        Cm.W[0][2][0] = -ONE / 3.0;
        Cm.W[0][2][1] = TWO / 3.0;
        Cm.W[0][3][1] = -TWO / 3.0;
        Cm.W[0][3][2] = ONE;
        Cm.W[0][4][0] = ONE / 12.0;
        Cm.W[0][4][1] = -ONE / 3.0;
        Cm.W[0][4][2] = 7.0 / 12.0;

        Cm.W[1][3][0] = ONE / TWO;
        Cm.W[1][3][2] = -ONE / TWO;
    }
    Some(C)
}

fn coeff_ARKODE_MRI_GARK_RALSTON3() -> Option<MRIStepCoupling> {
    /* Roberts et al., SISC 44:A1405 - A1427, 2022 */
    let C = MRIStepCoupling_Alloc(2, 4, MRISTEP_EXPLICIT)?;
    {
        let mut Cm = C.borrow_mut();

        Cm.q = 3;
        Cm.p = 2;

        Cm.c[1] = ONE / TWO;
        Cm.c[2] = 3.0 / 4.0;
        Cm.c[3] = ONE;

        Cm.W[0][1][0] = ONE / TWO;
        Cm.W[0][2][0] = -11.0 / 4.0;
        Cm.W[0][2][1] = 3.0;
        Cm.W[0][3][0] = 47.0 / 36.0;
        Cm.W[0][3][1] = -ONE / 6.0;
        Cm.W[0][3][2] = -8.0 / 9.0;
        Cm.W[0][4][0] = ONE / 40.0;
        Cm.W[0][4][1] = 7.0 / 40.0;
        Cm.W[0][4][2] = ONE / 20.0;

        Cm.W[1][2][0] = 9.0 / TWO;
        Cm.W[1][2][1] = -9.0 / TWO;
        Cm.W[1][3][0] = -13.0 / 6.0;
        Cm.W[1][3][1] = -ONE / TWO;
        Cm.W[1][3][2] = 8.0 / 3.0;
    }
    Some(C)
}

fn coeff_ARKODE_MRI_GARK_ERK45a() -> Option<MRIStepCoupling> {
    /* A. Sandu, SINUM 57:2300-2327, 2019 */
    /* Embedding coefficients CORRECTED in A. Sandu, arxiv:1808.02759, 2018 */
    let C = MRIStepCoupling_Alloc(2, 6, MRISTEP_EXPLICIT)?;
    {
        let mut Cm = C.borrow_mut();

        Cm.q = 4;
        Cm.p = 3;

        Cm.c[1] = 0.2;
        Cm.c[2] = 0.4;
        Cm.c[3] = 0.6;
        Cm.c[4] = 0.8;
        Cm.c[5] = ONE;

        Cm.W[0][1][0] = 0.2;
        Cm.W[0][2][0] = -53.0 / 16.0;
        Cm.W[0][2][1] = 281.0 / 80.0;
        Cm.W[0][3][0] = -36562993.0 / 71394880.0;
        Cm.W[0][3][1] = 34903117.0 / 17848720.0;
        Cm.W[0][3][2] = -88770499.0 / 71394880.0;
        Cm.W[0][4][0] = -7631593.0 / 71394880.0;
        Cm.W[0][4][1] = -166232021.0 / 35697440.0;
        Cm.W[0][4][2] = 6068517.0 / 1519040.0;
        Cm.W[0][4][3] = 8644289.0 / 8924360.0;
        Cm.W[0][5][0] = 277061.0 / 303808.0;
        Cm.W[0][5][1] = -209323.0 / 1139280.0;
        Cm.W[0][5][2] = -1360217.0 / 1139280.0;
        Cm.W[0][5][3] = -148789.0 / 56964.0;
        Cm.W[0][5][4] = 147889.0 / 45120.0;
        Cm.W[0][6][0] = -88227.0 / 47470.0;
        Cm.W[0][6][1] = 756870829.0 / 340217490.0;
        Cm.W[0][6][2] = -713704111.0 / 1360869960.0;
        Cm.W[0][6][3] = -31967827.0 / 340217490.0;
        Cm.W[0][6][4] = 129673.0 / 286680.0;

        Cm.W[1][2][0] = 503.0 / 80.0;
        Cm.W[1][2][1] = -503.0 / 80.0;
        Cm.W[1][3][0] = -1365537.0 / 35697440.0;
        Cm.W[1][3][1] = 4963773.0 / 7139488.0;
        Cm.W[1][3][2] = -1465833.0 / 2231090.0;
        Cm.W[1][4][0] = 66974357.0 / 35697440.0;
        Cm.W[1][4][1] = 21445367.0 / 7139488.0;
        Cm.W[1][4][2] = -3.0;
        Cm.W[1][4][3] = -8388609.0 / 4462180.0;
        Cm.W[1][5][0] = -18227.0 / 7520.0;
        Cm.W[1][5][1] = TWO;
        Cm.W[1][5][2] = ONE;
        Cm.W[1][5][3] = 5.0;
        Cm.W[1][5][4] = -41933.0 / 7520.0;
        Cm.W[1][6][0] = 6213.0 / 1880.0;
        Cm.W[1][6][1] = -6213.0 / 1880.0;
    }
    Some(C)
}

fn coeff_ARKODE_MRI_GARK_BACKWARD_EULER() -> Option<MRIStepCoupling> {
    let C = MRIStepCoupling_Alloc(1, 3, MRISTEP_IMPLICIT)?;
    {
        let mut Cm = C.borrow_mut();

        Cm.q = 1;
        Cm.p = 0;

        Cm.c[1] = ONE;
        Cm.c[2] = ONE;

        Cm.G[0][1][0] = ONE;
        Cm.G[0][2][0] = -ONE;
        Cm.G[0][2][2] = ONE;
    }
    Some(C)
}

fn coeff_ARKODE_MRI_GARK_IRK21a() -> Option<MRIStepCoupling> {
    /* A. Sandu, SINUM 57:2300-2327, 2019 */
    let B = ARKodeButcherTable_Alloc(3, SUNTRUE)?;
    {
        let mut Bm = B.borrow_mut();

        Bm.q = 2;
        Bm.p = 1;

        Bm.c[1] = ONE;
        Bm.c[2] = ONE;

        Bm.A[1][0] = ONE;
        Bm.A[2][0] = 0.5;
        Bm.A[2][2] = 0.5;

        Bm.b[0] = 0.5;
        Bm.b[2] = 0.5;

        Bm.d[2] = 1.0;
    }

    let (q, p) = {
        let Bm = B.borrow();
        (Bm.q, Bm.p)
    };
    let C = MRIStepCoupling_MIStoMRI(Some(&B), q, p);
    drop(B); /* ARKodeButcherTable_Free(B) */
    C
}

fn coeff_ARKODE_MRI_GARK_IMPLICIT_MIDPOINT() -> Option<MRIStepCoupling> {
    let C = MRIStepCoupling_Alloc(1, 4, MRISTEP_IMPLICIT)?;
    {
        let mut Cm = C.borrow_mut();

        Cm.q = 2;
        Cm.p = 0;

        Cm.c[1] = ONE / TWO;
        Cm.c[2] = ONE / TWO;
        Cm.c[3] = ONE;

        Cm.G[0][1][0] = ONE / TWO;
        Cm.G[0][2][0] = -ONE / TWO;
        Cm.G[0][2][2] = ONE / TWO;
        Cm.G[0][3][2] = ONE / TWO;
    }
    Some(C)
}

fn coeff_ARKODE_MRI_GARK_ESDIRK34a() -> Option<MRIStepCoupling> {
    /* A. Sandu, SINUM 57:2300-2327, 2019 */
    let C = MRIStepCoupling_Alloc(1, 8, MRISTEP_IMPLICIT)?;
    let beta: sunrealtype = 0.4358665215084589994160194511935568425;
    {
        let mut Cm = C.borrow_mut();

        Cm.q = 3;
        Cm.p = 2;

        Cm.c[1] = ONE / 3.0;
        Cm.c[2] = ONE / 3.0;
        Cm.c[3] = TWO / 3.0;
        Cm.c[4] = TWO / 3.0;
        Cm.c[5] = ONE;
        Cm.c[6] = ONE;
        Cm.c[7] = ONE;

        Cm.G[0][1][0] = ONE / 3.0;
        Cm.G[0][2][0] = -beta;
        Cm.G[0][2][2] = beta;
        Cm.G[0][3][0] = -0.3045790611944504970424837655380884888;
        Cm.G[0][3][2] = 0.6379123945277838303758170988714218222;
        Cm.G[0][4][0] = 0.2116913105640266601676536489364004869;
        Cm.G[0][4][2] = -0.6475578320724856595836731001299573294;
        Cm.G[0][4][4] = beta;
        Cm.G[0][5][0] = 0.4454209388055495029575162344619115112;
        Cm.G[0][5][2] = 0.8813784805616198280398949036456491923;
        Cm.G[0][5][4] = -0.9934660860338359976640778047742273701;
        Cm.G[0][6][0] = -beta;
        Cm.G[0][6][6] = beta;
        Cm.G[0][8][0] = 0.2453831999117524372455680781104585876241;
        Cm.G[0][8][2] = 0.4204215033044044563073464989473988121422;
        Cm.G[0][8][4] = -1.576992606344066224351397232226173387157;
        Cm.G[0][8][6] = 0.9111879031279093307984826551683159873903;
    }
    Some(C)
}

fn coeff_ARKODE_MRI_GARK_ESDIRK46a() -> Option<MRIStepCoupling> {
    /* A. Sandu, SINUM 57:2300-2327, 2019 */
    let C = MRIStepCoupling_Alloc(2, 12, MRISTEP_IMPLICIT)?;
    {
        let mut Cm = C.borrow_mut();

        Cm.q = 4;
        Cm.p = 3;

        Cm.c[1] = ONE / 5.0;
        Cm.c[2] = ONE / 5.0;
        Cm.c[3] = TWO / 5.0;
        Cm.c[4] = TWO / 5.0;
        Cm.c[5] = 3.0 / 5.0;
        Cm.c[6] = 3.0 / 5.0;
        Cm.c[7] = 4.0 / 5.0;
        Cm.c[8] = 4.0 / 5.0;
        Cm.c[9] = ONE;
        Cm.c[10] = ONE;
        Cm.c[11] = ONE;

        Cm.G[0][1][0] = ONE / 5.0;
        Cm.G[0][2][0] = -ONE / 4.0;
        Cm.G[0][2][2] = ONE / 4.0;
        Cm.G[0][3][0] = 1771023115159.0 / 1929363690800.0;
        Cm.G[0][3][2] = -1385150376999.0 / 1929363690800.0;
        Cm.G[0][4][0] = 914009.0 / 345800.0;
        Cm.G[0][4][2] = -1000459.0 / 345800.0;
        Cm.G[0][4][4] = ONE / 4.0;
        Cm.G[0][5][0] = 18386293581909.0 / 36657910125200.0;
        Cm.G[0][5][2] = 5506531089.0 / 80566835440.0;
        Cm.G[0][5][4] = -178423463189.0 / 482340922700.0;
        Cm.G[0][6][0] = 36036097.0 / 8299200.0;
        Cm.G[0][6][2] = 4621.0 / 118560.0;
        Cm.G[0][6][4] = -38434367.0 / 8299200.0;
        Cm.G[0][6][6] = ONE / 4.0;
        Cm.G[0][7][0] = -247809665162987.0 / 146631640500800.0;
        Cm.G[0][7][2] = 10604946373579.0 / 14663164050080.0;
        Cm.G[0][7][4] = 10838126175385.0 / 5865265620032.0;
        Cm.G[0][7][6] = -24966656214317.0 / 36657910125200.0;
        Cm.G[0][8][0] = 38519701.0 / 11618880.0;
        Cm.G[0][8][2] = 10517363.0 / 9682400.0;
        Cm.G[0][8][4] = -23284701.0 / 19364800.0;
        Cm.G[0][8][6] = -10018609.0 / 2904720.0;
        Cm.G[0][8][8] = ONE / 4.0;
        Cm.G[0][9][0] = -52907807977903.0 / 33838070884800.0;
        Cm.G[0][9][2] = 74846944529257.0 / 73315820250400.0;
        Cm.G[0][9][4] = 365022522318171.0 / 146631640500800.0;
        Cm.G[0][9][6] = -20513210406809.0 / 109973730375600.0;
        Cm.G[0][9][8] = -2918009798.0 / 1870301537.0;
        Cm.G[0][10][0] = 19.0 / 100.0;
        Cm.G[0][10][2] = -73.0 / 300.0;
        Cm.G[0][10][4] = 127.0 / 300.0;
        Cm.G[0][10][6] = 127.0 / 300.0;
        Cm.G[0][10][8] = -313.0 / 300.0;
        Cm.G[0][10][10] = ONE / 4.0;
        Cm.G[0][12][0] = -ONE / 4.0;
        Cm.G[0][12][2] = 5595.0 / 8804.0;
        Cm.G[0][12][4] = -2445.0 / 8804.0;
        Cm.G[0][12][6] = -4225.0 / 8804.0;
        Cm.G[0][12][8] = 2205.0 / 4402.0;
        Cm.G[0][12][10] = -567.0 / 4402.0;

        Cm.G[1][3][0] = -1674554930619.0 / 964681845400.0;
        Cm.G[1][3][2] = 1674554930619.0 / 964681845400.0;
        Cm.G[1][4][0] = -1007739.0 / 172900.0;
        Cm.G[1][4][2] = 1007739.0 / 172900.0;
        Cm.G[1][5][0] = -8450070574289.0 / 18328955062600.0;
        Cm.G[1][5][2] = -39429409169.0 / 40283417720.0;
        Cm.G[1][5][4] = 173621393067.0 / 120585230675.0;
        Cm.G[1][6][0] = -122894383.0 / 16598400.0;
        Cm.G[1][6][2] = 14501.0 / 237120.0;
        Cm.G[1][6][4] = 121879313.0 / 16598400.0;
        Cm.G[1][7][0] = 32410002731287.0 / 15434909526400.0;
        Cm.G[1][7][2] = -46499276605921.0 / 29326328100160.0;
        Cm.G[1][7][4] = -34914135774643.0 / 11730531240064.0;
        Cm.G[1][7][6] = 45128506783177.0 / 18328955062600.0;
        Cm.G[1][8][0] = -128357303.0 / 23237760.0;
        Cm.G[1][8][2] = -35433927.0 / 19364800.0;
        Cm.G[1][8][4] = 71038479.0 / 38729600.0;
        Cm.G[1][8][6] = 8015933.0 / 1452360.0;
        Cm.G[1][9][0] = 136721604296777.0 / 67676141769600.0;
        Cm.G[1][9][2] = -349632444539303.0 / 146631640500800.0;
        Cm.G[1][9][4] = -1292744859249609.0 / 293263281001600.0;
        Cm.G[1][9][6] = 8356250416309.0 / 54986865187800.0;
        Cm.G[1][9][8] = 17282943803.0 / 3740603074.0;
        Cm.G[1][10][0] = 3.0 / 25.0;
        Cm.G[1][10][2] = -29.0 / 300.0;
        Cm.G[1][10][4] = 71.0 / 300.0;
        Cm.G[1][10][6] = 71.0 / 300.0;
        Cm.G[1][10][8] = -149.0 / 300.0;
    }
    Some(C)
}

fn coeff_ARKODE_IMEX_MRI_GARK_ASCHER_ARK2() -> Option<MRIStepCoupling> {
    let C = MRIStepCoupling_Alloc(1, 5, MRISTEP_IMEX)?;

    let gamma: sunrealtype = ONE - ONE / SUNRsqrt(2.0);
    let delta: sunrealtype = ONE - ONE / (2.0 * gamma);
    {
        let mut Cm = C.borrow_mut();

        Cm.q = 2;
        Cm.p = 1;

        Cm.c[1] = gamma;
        Cm.c[2] = gamma;
        Cm.c[3] = ONE;
        Cm.c[4] = ONE;

        Cm.W[0][1][0] = gamma;
        Cm.W[0][3][0] = delta - gamma;
        Cm.W[0][3][2] = ONE - delta;
        Cm.W[0][5][0] = -delta;
        Cm.W[0][5][2] = delta - 0.4;
        Cm.W[0][5][4] = 0.4;

        Cm.G[0][1][0] = gamma;
        Cm.G[0][2][0] = -gamma;
        Cm.G[0][2][2] = gamma;
        Cm.G[0][3][2] = ONE - gamma;
        Cm.G[0][4][2] = -gamma;
        Cm.G[0][4][4] = gamma;
        Cm.G[0][5][2] = -0.4;
        Cm.G[0][5][4] = 0.4;
    }
    Some(C)
}

fn coeff_ARKODE_IMEX_MRI_GARK_ARK2() -> Option<MRIStepCoupling> {
    let C = MRIStepCoupling_Alloc(1, 6, MRISTEP_IMEX)?;
    {
        let mut Cm = C.borrow_mut();

        Cm.q = 2;
        Cm.p = 1;

        Cm.c[1] = 0.5857864376269049511983112757903;
        Cm.c[2] = 0.5857864376269049511983112757903;
        Cm.c[3] = ONE;
        Cm.c[4] = ONE;
        Cm.c[5] = ONE;

        Cm.W[0][1][0] = 0.5857864376269049511983112757903; /* 2 - sqrt(2) */
        Cm.W[0][3][0] = -0.55719095841793663413220751719353; /* (3 - 2 * sqrt(2))/6 - (2 - sqrt(2)) */
        Cm.W[0][3][2] = 0.97140452079103168293389624140323; /* (3 + 2 * sqrt(2))/6 */
        Cm.W[0][5][0] = 0.32495791138430544513431842245566; /* 1/(2 * sqrt(2)) - (3 - 2 * sqrt(2))/6 */
        Cm.W[0][5][2] = -0.61785113019775792073347406035081; /* 1/(2 * sqrt(2)) - (3 + 2 * sqrt(2))/6 */
        Cm.W[0][5][4] = 0.29289321881345247559915563789515; /* 1 - 1 / sqrt(2)) */
        Cm.W[0][6][0] = 0.29462782549439480183368515087702; /* (4 - sqrt(2)) / 8 - (3 - 2 * sqrt(2))/6 */
        Cm.W[0][6][2] = -0.64818121608766856403410733192944; /* (4 - sqrt(2)) / 8 - (3 + 2 * sqrt(2))/6 */
        Cm.W[0][6][4] = 0.35355339059327376220042218105242; /* 1 / (2 * sqrt(2)) */

        Cm.G[0][1][0] = 0.5857864376269049511983112757903; /* 2 - sqrt(2) */
        Cm.G[0][2][0] = -0.29289321881345247559915563789515; /* 1 - 1 / sqrt(2) - (2 - sqrt(2)) */
        Cm.G[0][2][2] = 0.29289321881345247559915563789515; /* 1 - 1 / sqrt(2) */
        Cm.G[0][3][0] = -0.29289321881345247559915563789515; /* 1 / sqrt(2) - 1 */
        Cm.G[0][3][2] = 0.70710678118654752440084436210485; /* 1 / sqrt(2) */
        Cm.G[0][4][0] = 0.35355339059327376220042218105242; /* 1 / (2 * sqrt(2)) */
        Cm.G[0][4][2] = -0.64644660940672623779957781894758; /* 1 / (2 * sqrt(2)) - 1 */
        Cm.G[0][4][4] = 0.29289321881345247559915563789515; /* 1 - 1 / sqrt(2) */
        Cm.G[0][6][0] = -0.030330085889910643300633271578637; /* (4 - sqrt(2)) / 8 - 1 / (2 * sqrt(2)) */
        Cm.G[0][6][2] = -0.030330085889910643300633271578637; /* (4 - sqrt(2)) / 8 - 1 / (2 * sqrt(2)) */
        Cm.G[0][6][4] = 0.060660171779821286601266543157274; /* 1 / (2 * sqrt(2)) - (1 - 1 / sqrt(2)) */
    }
    Some(C)
}

fn coeff_ARKODE_IMEX_MRI_GARK_EULER() -> Option<MRIStepCoupling> {
    let C = MRIStepCoupling_Alloc(1, 3, MRISTEP_IMEX)?;
    {
        let mut Cm = C.borrow_mut();

        Cm.q = 1;
        Cm.p = 0;

        Cm.c[1] = ONE;
        Cm.c[2] = ONE;

        Cm.W[0][1][0] = ONE;

        Cm.G[0][1][0] = ONE;
        Cm.G[0][2][0] = -ONE;
        Cm.G[0][2][2] = ONE;
    }
    Some(C)
}

fn coeff_ARKODE_IMEX_MRI_GARK_TRAPEZOIDAL() -> Option<MRIStepCoupling> {
    let C = MRIStepCoupling_Alloc(1, 4, MRISTEP_IMEX)?;
    {
        let mut Cm = C.borrow_mut();

        Cm.q = 2;
        Cm.p = 0;

        Cm.c[1] = ONE;
        Cm.c[2] = ONE;
        Cm.c[3] = ONE;

        Cm.W[0][1][0] = ONE;
        Cm.W[0][3][0] = -ONE / TWO;
        Cm.W[0][3][2] = ONE / TWO;

        Cm.G[0][1][0] = ONE;
        Cm.G[0][2][0] = -ONE / TWO;
        Cm.G[0][2][2] = ONE / TWO;
    }
    Some(C)
}

fn coeff_ARKODE_IMEX_MRI_GARK_MIDPOINT() -> Option<MRIStepCoupling> {
    let C = MRIStepCoupling_Alloc(1, 4, MRISTEP_IMEX)?;
    {
        let mut Cm = C.borrow_mut();

        Cm.q = 2;
        Cm.p = 0;

        Cm.c[1] = ONE / TWO;
        Cm.c[2] = ONE / TWO;
        Cm.c[3] = ONE;

        Cm.W[0][1][0] = ONE / TWO;
        Cm.W[0][3][0] = -ONE / TWO;
        Cm.W[0][3][2] = ONE;

        Cm.G[0][1][0] = ONE / TWO;
        Cm.G[0][2][0] = -ONE / TWO;
        Cm.G[0][2][2] = ONE / TWO;
        Cm.G[0][3][2] = ONE / TWO;
    }
    Some(C)
}

fn coeff_ARKODE_IMEX_MRI_GARK3a() -> Option<MRIStepCoupling> {
    /* R. Chinomona & D. Reynolds SINUM 43(5):A3082-A3113, 2021 */
    let C = MRIStepCoupling_Alloc(1, 8, MRISTEP_IMEX)?;
    let beta: sunrealtype = 0.4358665215084589994160194511935568425;
    {
        let mut Cm = C.borrow_mut();

        Cm.q = 3;
        Cm.p = 0;

        Cm.c[1] = beta;
        Cm.c[2] = beta;
        Cm.c[3] = 0.7179332607542294997080097255967784213;
        Cm.c[4] = 0.7179332607542294997080097255967784213;
        Cm.c[5] = ONE;
        Cm.c[6] = ONE;
        Cm.c[7] = ONE;

        Cm.W[0][1][0] = beta;
        Cm.W[0][3][0] = -0.5688715801234400928465032925317932021;
        Cm.W[0][3][2] = 0.8509383193692105931384935669350147809;
        Cm.W[0][4][0] = 0.454283944643608855878770886900124654;
        Cm.W[0][4][2] = -0.454283944643608855878770886900124654;
        Cm.W[0][5][0] = -0.4271371821005074011706645050390732474;
        Cm.W[0][5][2] = 0.1562747733103380821014660497037023496;
        Cm.W[0][5][4] = 0.5529291480359398193611887297385924765;
        Cm.W[0][7][0] = 0.105858296071879638722377459477184953;
        Cm.W[0][7][2] = 0.655567501140070250975288954324730635;
        Cm.W[0][7][4] = -1.197292318720408889113685864995472431;
        Cm.W[0][7][6] = beta;

        Cm.G[0][1][0] = beta;
        Cm.G[0][2][0] = -beta;
        Cm.G[0][2][2] = beta;
        Cm.G[0][3][0] = -0.4103336962288525014599513720161078937;
        Cm.G[0][3][2] = 0.6924004354746230017519416464193294724;
        Cm.G[0][4][0] = 0.4103336962288525014599513720161078937;
        Cm.G[0][4][2] = -0.8462002177373115008759708232096647362;
        Cm.G[0][4][4] = beta;
        Cm.G[0][5][0] = beta;
        Cm.G[0][5][2] = 0.9264299099302395700444874096601015328;
        Cm.G[0][5][4] = -1.080229692192928069168516586450436797;
        Cm.G[0][6][0] = -beta;
        Cm.G[0][6][6] = beta;
    }
    Some(C)
}

fn coeff_ARKODE_IMEX_MRI_GARK3b() -> Option<MRIStepCoupling> {
    /* R. Chinomona & D. Reynolds SINUM 43(5):A3082-A3113, 2021 */
    let C = MRIStepCoupling_Alloc(1, 8, MRISTEP_IMEX)?;
    let beta: sunrealtype = 0.4358665215084589994160194511935568425;
    {
        let mut Cm = C.borrow_mut();

        Cm.q = 3;
        Cm.p = 0;

        Cm.c[1] = beta;
        Cm.c[2] = beta;
        Cm.c[3] = 0.7179332607542294997080097255967784213;
        Cm.c[4] = 0.7179332607542294997080097255967784213;
        Cm.c[5] = ONE;
        Cm.c[6] = ONE;
        Cm.c[7] = ONE;

        Cm.W[0][1][0] = beta;
        Cm.W[0][3][0] = -0.1750145285570467590610670000018749059;
        Cm.W[0][3][2] = 0.4570812678028172593530572744050964846;
        Cm.W[0][4][0] = 0.06042689307721552209333459437020635774;
        Cm.W[0][4][2] = -0.06042689307721552209333459437020635774;
        Cm.W[0][5][0] = 0.1195213959425454440038786034027936869;
        Cm.W[0][5][2] = -1.84372522668966191789853395029629765;
        Cm.W[0][5][4] = 2.006270569992886974186645621296725542;
        Cm.W[0][6][0] = -0.5466585780430528451745431084418669343;
        Cm.W[0][6][2] = 2.0;
        Cm.W[0][6][4] = -1.453341421956947154825456891558133066;
        Cm.W[0][7][0] = 0.105858296071879638722377459477184953;
        Cm.W[0][7][2] = 0.655567501140070250975288954324730635;
        Cm.W[0][7][4] = -1.197292318720408889113685864995472431;
        Cm.W[0][7][6] = beta;

        Cm.G[0][1][0] = beta;
        Cm.G[0][2][0] = -beta;
        Cm.G[0][2][2] = beta;
        Cm.G[0][3][0] = 0.0414273753564414837153799230278275639;
        Cm.G[0][3][2] = 0.2406393638893290165766103513753940148;
        Cm.G[0][4][0] = -0.0414273753564414837153799230278275639;
        Cm.G[0][4][2] = -0.3944391461520175157006395281657292786;
        Cm.G[0][4][4] = beta;
        Cm.G[0][5][0] = 0.1123373143006047802633543416889605123;
        Cm.G[0][5][2] = 1.051807513648115027700693049638099167;
        Cm.G[0][5][4] = -0.8820780887029493076720571169238381009;
        Cm.G[0][6][0] = -0.1123373143006047802633543416889605123;
        Cm.G[0][6][2] = -0.1253776037178754576562056399779976346;
        Cm.G[0][6][4] = -0.1981516034899787614964594695265986957;
        Cm.G[0][6][6] = beta;
    }
    Some(C)
}

fn coeff_ARKODE_IMEX_MRI_GARK4() -> Option<MRIStepCoupling> {
    /* R. Chinomona & D. Reynolds SINUM 43(5):A3082-A3113, 2021 */
    let C = MRIStepCoupling_Alloc(2, 12, MRISTEP_IMEX)?;
    {
        let mut Cm = C.borrow_mut();

        Cm.q = 4;
        Cm.p = 0;

        Cm.c[1] = 0.5;
        Cm.c[2] = 0.5;
        Cm.c[3] = 0.625;
        Cm.c[4] = 0.625;
        Cm.c[5] = 0.75;
        Cm.c[6] = 0.75;
        Cm.c[7] = 0.875;
        Cm.c[8] = 0.875;
        Cm.c[9] = ONE;
        Cm.c[10] = ONE;
        Cm.c[11] = ONE;

        Cm.W[0][1][0] = 0.5;
        Cm.W[0][3][0] = -1.91716534363662868878172216064946905;
        Cm.W[0][3][2] = 2.04216534363662868878172216064946905;
        Cm.W[0][4][0] = -0.4047510318011059426979159070469904691;
        Cm.W[0][4][2] = 0.4047510318011059426979159070469904691;
        Cm.W[0][5][0] = 11.45146602249221636665698028602631728;
        Cm.W[0][5][2] = -30.21075747526504271440647815573950607;
        Cm.W[0][5][4] = 18.88429145277282634774949786971318879;
        Cm.W[0][6][0] = -0.7090335647602614506847116729463301439;
        Cm.W[0][6][2] = 1.03030720858751876652616190884004718;
        Cm.W[0][6][4] = -0.3212736438272573158414502358937170357;
        Cm.W[0][7][0] = -29.99548716455828439840910684944199275;
        Cm.W[0][7][2] = 37.60598277499180180536489685624385701;
        Cm.W[0][7][4] = 0.3212736438272573158414502358937170357;
        Cm.W[0][7][6] = -7.806769254260774722797240242695581295;
        Cm.W[0][8][0] = 3.104665054272962116338769391849124223;
        Cm.W[0][8][2] = -2.430325019757162297132065927415566359;
        Cm.W[0][8][4] = -1.905479301151524635219201659483842131;
        Cm.W[0][8][6] = 1.231139266635724816012498195050284266;
        Cm.W[0][9][0] = -2.424429547752047869875875914355514008;
        Cm.W[0][9][2] = 2.430325019757162297132065927415566359;
        Cm.W[0][9][4] = 1.905479301151524635219201659483842131;
        Cm.W[0][9][6] = -1.231139266635724816012498195050284266;
        Cm.W[0][9][8] = -0.555235506520914246462893477493610215;
        Cm.W[0][10][0] = -0.01044135044479748590294518945165354204;
        Cm.W[0][10][2] = 0.07260303614655074505152104505488141613;
        Cm.W[0][10][4] = -0.1288275951677260952239454098576424313;
        Cm.W[0][10][6] = 0.1129355350093823566139440107122154084;
        Cm.W[0][10][8] = -0.04626962554340952053857445645780085125;
        Cm.W[0][11][0] = -0.8108522787762101328175789228607932098;
        Cm.W[0][11][2] = 0.2560073199220492435001562192140882299;
        Cm.W[0][11][4] = 0.8068294072697527893665866422787819475;
        Cm.W[0][11][6] = -0.4557148228721823795105894821742761164;
        Cm.W[0][11][8] = -0.04626962554340952053857445645780085125;
        Cm.W[0][11][10] = 0.25;

        Cm.W[1][3][0] = 4.084330687273257377563444321298938099;
        Cm.W[1][3][2] = -4.084330687273257377563444321298938099;
        Cm.W[1][5][0] = -21.84342998138222084791812875795865363;
        Cm.W[1][5][2] = 59.61201288692787354341712449738503121;
        Cm.W[1][5][4] = -37.76858290554565269549899573942637758;
        Cm.W[1][7][0] = 61.65904145863709169818763704477664579;
        Cm.W[1][7][2] = -77.27257996715864114378211753016780838;
        Cm.W[1][7][6] = 15.61353850852154944559448048539116259;
        Cm.W[1][9][0] = -1.11047101304182849292578695498722043;
        Cm.W[1][9][8] = 1.11047101304182849292578695498722043;

        Cm.G[0][1][0] = 0.5;
        Cm.G[0][2][0] = -0.25;
        Cm.G[0][2][2] = 0.25;
        Cm.G[0][3][0] = -3.977281248108488183067033851462278892;
        Cm.G[0][3][2] = 4.102281248108488183067033851462278892;
        Cm.G[0][4][0] = -0.06905388741401691232724147084809374064;
        Cm.G[0][4][2] = -0.1809461125859830876727585291519062594;
        Cm.G[0][4][4] = 0.25;
        Cm.G[0][5][0] = -1.761767663757920528863378964822412405;
        Cm.G[0][5][2] = 2.694524698377298610155338150791461384;
        Cm.G[0][5][4] = -0.8077570346193780812919591859690489783;
        Cm.G[0][6][0] = 0.5558721791553969487305081009588084962;
        Cm.G[0][6][2] = -0.6799140501579995013958501527883486949;
        Cm.G[0][6][4] = -0.1259581289973974473346579481704598013;
        Cm.G[0][6][6] = 0.25;
        Cm.G[0][7][0] = -5.840176028724955954446426657541065113;
        Cm.G[0][7][2] = 8.174456684291915089191270805710716374;
        Cm.G[0][7][4] = 0.1259581289973974473346579481704598013;
        Cm.G[0][7][6] = -2.335238784564356582079502096340111063;
        Cm.G[0][8][0] = -1.906792645167811808094759305036052304;
        Cm.G[0][8][2] = -1.547057811385123933632984579249388443;
        Cm.G[0][8][4] = 4.129888013149350305954491738020313225;
        Cm.G[0][8][6] = -0.9260375565964145642267478537348724775;
        Cm.G[0][8][8] = 0.25;
        Cm.G[0][9][0] = 3.337028151688726054557652782529662519;
        Cm.G[0][9][2] = 1.547057811385123933632984579249388443;
        Cm.G[0][9][4] = -4.129888013149350305954491738020313225;
        Cm.G[0][9][6] = 0.9260375565964145642267478537348724775;
        Cm.G[0][9][8] = -1.555235506520914246462893477493610215;
        Cm.G[0][10][0] = -0.8212936292210076187205241123124467518;
        Cm.G[0][10][2] = 0.328610356068599988551677264268969646;
        Cm.G[0][10][4] = 0.6780018121020266941426412324211395162;
        Cm.G[0][10][6] = -0.3427792878628000228966454714620607079;
        Cm.G[0][10][8] = -0.0925392510868190410771489129156017025;
        Cm.G[0][10][10] = 0.25;

        Cm.G[1][3][0] = 8.704562496216976366134067702924557783;
        Cm.G[1][3][2] = -8.704562496216976366134067702924557783;
        Cm.G[1][5][0] = 3.911643102343874882381240871341012292;
        Cm.G[1][5][2] = -5.027157171582631044965159243279110249;
        Cm.G[1][5][4] = 1.115514069238756162583918371938097957;
        Cm.G[1][7][0] = 10.81860769913911801143183711316451323;
        Cm.G[1][7][2] = -14.98908526826783117559084130584473536;
        Cm.G[1][7][6] = 4.170477569128713164159004192680222125;
        Cm.G[1][9][0] = -2.61047101304182849292578695498722043;
        Cm.G[1][9][8] = 2.61047101304182849292578695498722043;
    }
    Some(C)
}

fn coeff_ARKODE_IMEX_MRI_SR21() -> Option<MRIStepCoupling> {
    /* A.C. Fish, D.R. Reynolds, S.B. Roberts, JCAM 438:115534, 2024 */
    let C = MRIStepCoupling_Alloc(1, 4, MRISTEP_SR)?;
    {
        let mut Cm = C.borrow_mut();

        Cm.q = 2;
        Cm.p = 1;

        Cm.c[1] = 3.0 / 5.0;
        Cm.c[2] = 4.0 / 15.0;
        Cm.c[3] = ONE;

        Cm.W[0][1][0] = 3.0 / 5.0;
        Cm.W[0][2][0] = 14.0 / 165.0;
        Cm.W[0][2][1] = 2.0 / 11.0;
        Cm.W[0][3][0] = -13.0 / 54.0;
        Cm.W[0][3][1] = 137.0 / 270.0;
        Cm.W[0][3][2] = 11.0 / 15.0;
        Cm.W[0][4][0] = -0.25;
        Cm.W[0][4][1] = 0.5;
        Cm.W[0][4][2] = 0.75;

        Cm.G[0][1][0] = -11.0 / 23.0;
        Cm.G[0][1][1] = 11.0 / 23.0;
        Cm.G[0][2][0] = -6692.0 / 52371.0;
        Cm.G[0][2][1] = -18355.0 / 52371.0;
        Cm.G[0][2][2] = 11.0 / 23.0;
        Cm.G[0][3][0] = 11621.0 / 90666.0;
        Cm.G[0][3][1] = -215249.0 / 226665.0;
        Cm.G[0][3][2] = 17287.0 / 50370.0;
        Cm.G[0][3][3] = 11.0 / 23.0;
        Cm.G[0][4][0] = -31.0 / 12.0;
        Cm.G[0][4][1] = -ONE / 6.0;
        Cm.G[0][4][2] = 11.0 / 4.0;
    }
    Some(C)
}

fn coeff_ARKODE_IMEX_MRI_SR32() -> Option<MRIStepCoupling> {
    /* A.C. Fish, D.R. Reynolds, S.B. Roberts, JCAM 438:115534, 2024 */
    let C = MRIStepCoupling_Alloc(2, 5, MRISTEP_SR)?;
    {
        let mut Cm = C.borrow_mut();

        Cm.q = 3;
        Cm.p = 2;

        Cm.c[1] = 23.0 / 34.0;
        Cm.c[2] = 4.0 / 5.0;
        Cm.c[3] = 17.0 / 15.0;
        Cm.c[4] = ONE;

        Cm.W[0][1][0] = 23.0 / 34.0;
        Cm.W[0][2][0] = 71.0 / 70.0;
        Cm.W[0][2][1] = -3.0 / 14.0;
        Cm.W[0][3][0] = 124.0 / 1155.0;
        Cm.W[0][3][1] = 4.0 / 7.0;
        Cm.W[0][3][2] = 5.0 / 11.0;
        Cm.W[0][4][0] = 162181.0 / 187680.0;
        Cm.W[0][4][1] = 119.0 / 1380.0;
        Cm.W[0][4][2] = 11.0 / 32.0;
        Cm.W[0][4][3] = -5.0 / 17.0;
        Cm.W[0][5][0] = 76355.0 / 74834.0;
        Cm.W[0][5][1] = -46.0 / 31.0;
        Cm.W[0][5][2] = 67.0 / 34.0;
        Cm.W[0][5][3] = -36.0 / 71.0;

        Cm.W[1][2][0] = -14453.0 / 63825.0;
        Cm.W[1][2][1] = 14453.0 / 63825.0;
        Cm.W[1][3][0] = -2101267877.0 / 1206582300.0;
        Cm.W[1][3][1] = 2476735438.0 / 301645575.0;
        Cm.W[1][3][2] = -13575085.0 / 2098404.0;
        Cm.W[1][4][0] = -762580446799.0 / 588660102960.0;
        Cm.W[1][4][1] = 11083240219.0 / 4328383110.0;
        Cm.W[1][4][2] = -211274129.0 / 100368304.0;
        Cm.W[1][4][3] = 89562055.0 / 106641323.0;
        Cm.W[1][5][0] = -3732974.0 / 2278035.0;
        Cm.W[1][5][1] = 13857574.0 / 2278035.0;
        Cm.W[1][5][2] = -52.0 / 9.0;
        Cm.W[1][5][3] = 4.0 / 3.0;

        Cm.G[0][1][0] = -4.0 / 7.0;
        Cm.G[0][1][1] = 4.0 / 7.0;
        Cm.G[0][2][0] = -2707004.0 / 3127425.0;
        Cm.G[0][2][1] = 919904.0 / 3127425.0;
        Cm.G[0][2][2] = 4.0 / 7.0;
        Cm.G[0][3][0] = 852879271.0 / 703839675.0;
        Cm.G[0][3][1] = -1575000496.0 / 703839675.0;
        Cm.G[0][3][2] = 5.0 / 11.0;
        Cm.G[0][3][3] = 4.0 / 7.0;
        Cm.G[0][4][0] = 43136869.0 / 2019912118.0;
        Cm.G[0][4][1] = -73810600.0 / 1009956059.0;
        Cm.G[0][4][2] = -17653551.0 / 87822266.0;
        Cm.G[0][4][3] = -13993902.0 / 43911133.0;
        Cm.G[0][4][4] = 4.0 / 7.0;
        Cm.G[0][5][0] = -179.0 / 4140.0;
        Cm.G[0][5][1] = 799.0 / 14490.0;
        Cm.G[0][5][2] = ONE / 14.0;
        Cm.G[0][5][3] = -ONE / 12.0;
    }
    Some(C)
}

fn coeff_ARKODE_IMEX_MRI_SR43() -> Option<MRIStepCoupling> {
    /* A.C. Fish, D.R. Reynolds, S.B. Roberts, arXiv:2301.00865, 2023 */
    let C = MRIStepCoupling_Alloc(2, 7, MRISTEP_SR)?;
    {
        let mut Cm = C.borrow_mut();

        Cm.q = 4;
        Cm.p = 3;

        Cm.c[1] = ONE / 4.0;
        Cm.c[2] = 3.0 / 4.0;
        Cm.c[3] = 11.0 / 20.0;
        Cm.c[4] = ONE / 2.0;
        Cm.c[5] = ONE;
        Cm.c[6] = ONE;

        Cm.W[0][1][0] = ONE / 4.0;
        Cm.W[0][2][0] = 9.0 / 8.0;
        Cm.W[0][2][1] = -3.0 / 8.0;
        Cm.W[0][3][0] = 187.0 / 2340.0;
        Cm.W[0][3][1] = 7.0 / 9.0;
        Cm.W[0][3][2] = -4.0 / 13.0;
        Cm.W[0][4][0] = 64.0 / 165.0;
        Cm.W[0][4][1] = ONE / 6.0;
        Cm.W[0][4][2] = -3.0 / 5.0;
        Cm.W[0][4][3] = 6.0 / 11.0;
        Cm.W[0][5][0] = 1816283.0 / 549120.0;
        Cm.W[0][5][1] = -2.0 / 9.0;
        Cm.W[0][5][2] = -4.0 / 11.0;
        Cm.W[0][5][3] = -ONE / 6.0;
        Cm.W[0][5][4] = -2561809.0 / 1647360.0;
        Cm.W[0][6][1] = 7.0 / 11.0;
        Cm.W[0][6][2] = -2203.0 / 264.0;
        Cm.W[0][6][3] = 10825.0 / 792.0;
        Cm.W[0][6][4] = -85.0 / 12.0;
        Cm.W[0][6][5] = 841.0 / 396.0;
        Cm.W[0][7][0] = ONE / 400.0;
        Cm.W[0][7][1] = 49.0 / 12.0;
        Cm.W[0][7][2] = 43.0 / 6.0;
        Cm.W[0][7][3] = -7.0 / 10.0;
        Cm.W[0][7][4] = -85.0 / 12.0;
        Cm.W[0][7][5] = -2963.0 / 1200.0;

        Cm.W[1][2][0] = -11.0 / 4.0;
        Cm.W[1][2][1] = 11.0 / 4.0;
        Cm.W[1][3][0] = -1228.0 / 2925.0;
        Cm.W[1][3][1] = -92.0 / 225.0;
        Cm.W[1][3][2] = 808.0 / 975.0;
        Cm.W[1][4][0] = -2572.0 / 2805.0;
        Cm.W[1][4][1] = 167.0 / 255.0;
        Cm.W[1][4][2] = 199.0 / 136.0;
        Cm.W[1][4][3] = -1797.0 / 1496.0;
        Cm.W[1][5][0] = -1816283.0 / 274560.0;
        Cm.W[1][5][1] = 253.0 / 36.0;
        Cm.W[1][5][2] = -23.0 / 44.0;
        Cm.W[1][5][3] = 76.0 / 3.0;
        Cm.W[1][5][4] = -20775791.0 / 823680.0;
        Cm.W[1][6][1] = 107.0 / 132.0;
        Cm.W[1][6][2] = 1289.0 / 88.0;
        Cm.W[1][6][3] = -9275.0 / 792.0;
        Cm.W[1][6][5] = -371.0 / 99.0;
        Cm.W[1][7][0] = -ONE / 200.0;
        Cm.W[1][7][1] = -137.0 / 24.0;
        Cm.W[1][7][2] = -235.0 / 16.0;
        Cm.W[1][7][3] = 1237.0 / 80.0;
        Cm.W[1][7][5] = 2963.0 / 600.0;

        Cm.G[0][1][0] = -ONE / 4.0;
        Cm.G[0][1][1] = ONE / 4.0;
        Cm.G[0][2][0] = ONE / 4.0;
        Cm.G[0][2][1] = -ONE / 2.0;
        Cm.G[0][2][2] = ONE / 4.0;
        Cm.G[0][3][0] = 13.0 / 100.0;
        Cm.G[0][3][1] = -7.0 / 30.0;
        Cm.G[0][3][2] = -11.0 / 75.0;
        Cm.G[0][3][3] = ONE / 4.0;
        Cm.G[0][4][0] = 6.0 / 85.0;
        Cm.G[0][4][1] = -301.0 / 1360.0;
        Cm.G[0][4][2] = -99.0 / 544.0;
        Cm.G[0][4][3] = 45.0 / 544.0;
        Cm.G[0][4][4] = ONE / 4.0;
        Cm.G[0][5][1] = -9.0 / 4.0;
        Cm.G[0][5][2] = -19.0 / 48.0;
        Cm.G[0][5][3] = -75.0 / 16.0;
        Cm.G[0][5][4] = 85.0 / 12.0;
        Cm.G[0][5][5] = ONE / 4.0;
    }
    Some(C)
}

fn coeff_ARKODE_MERK21() -> Option<MRIStepCoupling> {
    /* A.C. Fish, D.R. Reynolds, S.B. Roberts, JCAM 438:115534, 2024;
       D.R. Reynolds, S. Amihere, D. Mitchell, V.T. Luan, JCAM, 2026 (embedding) */
    let C = MRIStepCoupling_Alloc(2, 3, MRISTEP_MERK)?;
    let c2: sunrealtype = 0.5;
    {
        let mut Cm = C.borrow_mut();

        Cm.q = 2;
        Cm.p = 1;
        Cm.ngroup = 2;
        Cm.group[0][0] = 1;
        Cm.group[0][1] = 3;
        Cm.group[1][0] = 2;

        Cm.c[1] = c2;
        Cm.c[2] = ONE;

        Cm.W[0][1][0] = ONE;
        Cm.W[0][2][0] = ONE;
        Cm.W[0][3][0] = ONE;

        Cm.W[1][2][0] = -ONE / c2;
        Cm.W[1][2][1] = ONE / c2;
    }
    Some(C)
}

fn coeff_ARKODE_MERK32() -> Option<MRIStepCoupling> {
    /* A.C. Fish, D.R. Reynolds, S.B. Roberts, JCAM 438:115534, 2024;
       D.R. Reynolds, S. Amihere, D. Mitchell, V.T. Luan, JCAM, 2026 (embedding) */
    let C = MRIStepCoupling_Alloc(2, 4, MRISTEP_MERK)?;
    let c2: sunrealtype = 0.5;
    {
        let mut Cm = C.borrow_mut();

        Cm.q = 3;
        Cm.p = 2;
        Cm.ngroup = 3;
        Cm.group[0][0] = 1;
        Cm.group[1][0] = 2;
        Cm.group[1][1] = 4;
        Cm.group[2][0] = 3;

        Cm.c[1] = c2;
        Cm.c[2] = 2.0 / 3.0;
        Cm.c[3] = ONE;

        Cm.W[0][1][0] = ONE;
        Cm.W[0][2][0] = ONE;
        Cm.W[0][3][0] = ONE;
        Cm.W[0][4][0] = ONE;

        Cm.W[1][2][0] = -ONE / c2;
        Cm.W[1][2][1] = ONE / c2;
        Cm.W[1][3][0] = -1.5;
        Cm.W[1][3][2] = 1.5;
        Cm.W[1][4][0] = -ONE / c2;
        Cm.W[1][4][1] = ONE / c2;
    }
    Some(C)
}

fn coeff_ARKODE_MERK43() -> Option<MRIStepCoupling> {
    /* A.C. Fish, D.R. Reynolds, S.B. Roberts, JCAM 438:115534, 2024;
       D.R. Reynolds, S. Amihere, D. Mitchell, V.T. Luan, JCAM, 2026 (embedding) */
    let C = MRIStepCoupling_Alloc(3, 7, MRISTEP_MERK)?;
    let c2: sunrealtype = 0.5;
    let c3: sunrealtype = 0.5;
    let c4: sunrealtype = ONE / 3.0;
    let c5: sunrealtype = 5.0 / 6.0;
    let c6: sunrealtype = ONE / 3.0;
    {
        let mut Cm = C.borrow_mut();

        Cm.q = 4;
        Cm.p = 3;
        Cm.ngroup = 4;
        Cm.group[0][0] = 1;
        Cm.group[1][0] = 3;
        Cm.group[1][1] = 2;
        Cm.group[2][0] = 5;
        Cm.group[2][1] = 4;
        Cm.group[2][2] = 7;
        Cm.group[3][0] = 6;

        Cm.c[1] = c2;
        Cm.c[2] = c3;
        Cm.c[3] = c4;
        Cm.c[4] = c5;
        Cm.c[5] = c6;
        Cm.c[6] = ONE;

        Cm.W[0][1][0] = ONE;
        Cm.W[0][2][0] = ONE;
        Cm.W[0][3][0] = ONE;
        Cm.W[0][4][0] = ONE;
        Cm.W[0][5][0] = ONE;
        Cm.W[0][6][0] = ONE;
        Cm.W[0][7][0] = ONE;

        Cm.W[1][2][0] = -ONE / c2;
        Cm.W[1][2][1] = ONE / c2;
        Cm.W[1][3][0] = -ONE / c2;
        Cm.W[1][3][1] = ONE / c2;
        Cm.W[1][4][0] = c4 / c3 / (c3 - c4) - c3 / c4 / (c3 - c4);
        Cm.W[1][4][2] = -c4 / c3 / (c3 - c4);
        Cm.W[1][4][3] = c3 / c4 / (c3 - c4);
        Cm.W[1][5][0] = c4 / c3 / (c3 - c4) - c3 / c4 / (c3 - c4);
        Cm.W[1][5][2] = -c4 / c3 / (c3 - c4);
        Cm.W[1][5][3] = c3 / c4 / (c3 - c4);
        Cm.W[1][6][0] = c6 / c5 / (c5 - c6) - c5 / c6 / (c5 - c6);
        Cm.W[1][6][4] = -c6 / c5 / (c5 - c6);
        Cm.W[1][6][5] = c5 / c6 / (c5 - c6);
        Cm.W[1][7][0] = c4 / c3 / (c3 - c4) - c3 / c4 / (c3 - c4);
        Cm.W[1][7][2] = -c4 / c3 / (c3 - c4);
        Cm.W[1][7][3] = c3 / c4 / (c3 - c4);

        Cm.W[2][4][0] = ONE / c4 / (c3 - c4) - ONE / c3 / (c3 - c4);
        Cm.W[2][4][2] = ONE / c3 / (c3 - c4);
        Cm.W[2][4][3] = -ONE / c4 / (c3 - c4);
        Cm.W[2][5][0] = ONE / c4 / (c3 - c4) - ONE / c3 / (c3 - c4);
        Cm.W[2][5][2] = ONE / c3 / (c3 - c4);
        Cm.W[2][5][3] = -ONE / c4 / (c3 - c4);
        Cm.W[2][6][0] = ONE / c6 / (c5 - c6) - ONE / c5 / (c5 - c6);
        Cm.W[2][6][4] = ONE / c5 / (c5 - c6);
        Cm.W[2][6][5] = -ONE / c6 / (c5 - c6);
        Cm.W[2][7][0] = ONE / c4 / (c3 - c4) - ONE / c3 / (c3 - c4);
        Cm.W[2][7][2] = ONE / c3 / (c3 - c4);
        Cm.W[2][7][3] = -ONE / c4 / (c3 - c4);
    }
    Some(C)
}

fn coeff_ARKODE_MERK54() -> Option<MRIStepCoupling> {
    /* A.C. Fish, D.R. Reynolds, S.B. Roberts, JCAM 438:115534, 2024;
       D.R. Reynolds, S. Amihere, D. Mitchell, V.T. Luan, JCAM, 2026 (embedding) */
    let C = MRIStepCoupling_Alloc(4, 11, MRISTEP_MERK)?;
    let c2: sunrealtype = 0.5;
    let c3: sunrealtype = 0.5;
    let c4: sunrealtype = ONE / 3.0;
    let c5: sunrealtype = 0.5;
    let c6: sunrealtype = ONE / 3.0;
    let c7: sunrealtype = 0.25;
    let c8: sunrealtype = 0.7;
    let c9: sunrealtype = 0.5;
    let c10: sunrealtype = 2.0 / 3.0;
    let a2: sunrealtype = ONE / c2;
    let a3: sunrealtype = c4 / c3 / (c4 - c3);
    let a4: sunrealtype = c3 / c4 / (c3 - c4);
    let a5: sunrealtype = c6 * c7 / c5 / (c5 - c6) / (c5 - c7);
    let a6: sunrealtype = c5 * c7 / c6 / (c6 - c5) / (c6 - c7);
    let a7: sunrealtype = c5 * c6 / c7 / (c7 - c5) / (c7 - c6);
    let a8: sunrealtype = c9 * c10 / c8 / (c8 - c9) / (c8 - c10);
    let a9: sunrealtype = c8 * c10 / c9 / (c9 - c8) / (c9 - c10);
    let a10: sunrealtype = c8 * c9 / c10 / (c10 - c8) / (c10 - c9);
    let b3: sunrealtype = ONE / c3 / (c3 - c4);
    let b4: sunrealtype = ONE / c4 / (c3 - c4);
    let b5: sunrealtype = (c6 + c7) / c5 / (c5 - c6) / (c5 - c7);
    let b6: sunrealtype = (c5 + c7) / c6 / (c6 - c5) / (c6 - c7);
    let b7: sunrealtype = (c5 + c6) / c7 / (c7 - c5) / (c7 - c6);
    let b8: sunrealtype = (c9 + c10) / c8 / (c8 - c9) / (c8 - c10);
    let b9: sunrealtype = (c8 + c10) / c9 / (c9 - c8) / (c9 - c10);
    let b10: sunrealtype = (c8 + c9) / c10 / (c10 - c8) / (c10 - c9);
    let g5: sunrealtype = ONE / c5 / (c5 - c6) / (c5 - c7);
    let g6: sunrealtype = ONE / c6 / (c6 - c5) / (c6 - c7);
    let g7: sunrealtype = ONE / c7 / (c7 - c5) / (c7 - c6);
    let g8: sunrealtype = ONE / c8 / (c8 - c9) / (c8 - c10);
    let g9: sunrealtype = ONE / c9 / (c9 - c8) / (c9 - c10);
    let g10: sunrealtype = ONE / c10 / (c10 - c8) / (c10 - c9);
    {
        let mut Cm = C.borrow_mut();

        Cm.q = 5;
        Cm.p = 4;
        Cm.ngroup = 5;
        Cm.group[0][0] = 1;
        Cm.group[1][0] = 3;
        Cm.group[1][1] = 2;
        Cm.group[2][0] = 6;
        Cm.group[2][1] = 5;
        Cm.group[2][2] = 4;
        Cm.group[3][0] = 8;
        Cm.group[3][1] = 9;
        Cm.group[3][2] = 7;
        Cm.group[3][3] = 11;
        Cm.group[4][0] = 10;

        Cm.c[1] = c2;
        Cm.c[2] = c3;
        Cm.c[3] = c4;
        Cm.c[4] = c5;
        Cm.c[5] = c6;
        Cm.c[6] = c7;
        Cm.c[7] = c8;
        Cm.c[8] = c9;
        Cm.c[9] = c10;
        Cm.c[10] = ONE;

        Cm.W[0][1][0] = ONE;
        Cm.W[0][2][0] = ONE;
        Cm.W[0][3][0] = ONE;
        Cm.W[0][4][0] = ONE;
        Cm.W[0][5][0] = ONE;
        Cm.W[0][6][0] = ONE;
        Cm.W[0][7][0] = ONE;
        Cm.W[0][8][0] = ONE;
        Cm.W[0][9][0] = ONE;
        Cm.W[0][10][0] = ONE;
        Cm.W[0][11][0] = ONE;

        Cm.W[1][2][0] = -a2;
        Cm.W[1][2][1] = a2;
        Cm.W[1][3][0] = -a2;
        Cm.W[1][3][1] = a2;
        Cm.W[1][4][0] = -(a3 + a4);
        Cm.W[1][4][2] = a3;
        Cm.W[1][4][3] = a4;
        Cm.W[1][5][0] = -(a3 + a4);
        Cm.W[1][5][2] = a3;
        Cm.W[1][5][3] = a4;
        Cm.W[1][6][0] = -(a3 + a4);
        Cm.W[1][6][2] = a3;
        Cm.W[1][6][3] = a4;
        Cm.W[1][7][0] = -(a5 + a6 + a7);
        Cm.W[1][7][4] = a5;
        Cm.W[1][7][5] = a6;
        Cm.W[1][7][6] = a7;
        Cm.W[1][8][0] = -(a5 + a6 + a7);
        Cm.W[1][8][4] = a5;
        Cm.W[1][8][5] = a6;
        Cm.W[1][8][6] = a7;
        Cm.W[1][9][0] = -(a5 + a6 + a7);
        Cm.W[1][9][4] = a5;
        Cm.W[1][9][5] = a6;
        Cm.W[1][9][6] = a7;
        Cm.W[1][10][0] = -(a8 + a9 + a10);
        Cm.W[1][10][7] = a8;
        Cm.W[1][10][8] = a9;
        Cm.W[1][10][9] = a10;
        Cm.W[1][11][0] = -(a5 + a6 + a7);
        Cm.W[1][11][4] = a5;
        Cm.W[1][11][5] = a6;
        Cm.W[1][11][6] = a7;

        Cm.W[2][4][0] = b4 - b3;
        Cm.W[2][4][2] = b3;
        Cm.W[2][4][3] = -b4;
        Cm.W[2][5][0] = b4 - b3;
        Cm.W[2][5][2] = b3;
        Cm.W[2][5][3] = -b4;
        Cm.W[2][6][0] = b4 - b3;
        Cm.W[2][6][2] = b3;
        Cm.W[2][6][3] = -b4;
        Cm.W[2][7][0] = b5 + b6 + b7;
        Cm.W[2][7][4] = -b5;
        Cm.W[2][7][5] = -b6;
        Cm.W[2][7][6] = -b7;
        Cm.W[2][8][0] = b5 + b6 + b7;
        Cm.W[2][8][4] = -b5;
        Cm.W[2][8][5] = -b6;
        Cm.W[2][8][6] = -b7;
        Cm.W[2][9][0] = b5 + b6 + b7;
        Cm.W[2][9][4] = -b5;
        Cm.W[2][9][5] = -b6;
        Cm.W[2][9][6] = -b7;
        Cm.W[2][10][0] = b8 + b9 + b10;
        Cm.W[2][10][7] = -b8;
        Cm.W[2][10][8] = -b9;
        Cm.W[2][10][9] = -b10;
        Cm.W[2][11][0] = b5 + b6 + b7;
        Cm.W[2][11][4] = -b5;
        Cm.W[2][11][5] = -b6;
        Cm.W[2][11][6] = -b7;

        Cm.W[3][7][0] = -(g5 + g6 + g7);
        Cm.W[3][7][4] = g5;
        Cm.W[3][7][5] = g6;
        Cm.W[3][7][6] = g7;
        Cm.W[3][8][0] = -(g5 + g6 + g7);
        Cm.W[3][8][4] = g5;
        Cm.W[3][8][5] = g6;
        Cm.W[3][8][6] = g7;
        Cm.W[3][9][0] = -(g5 + g6 + g7);
        Cm.W[3][9][4] = g5;
        Cm.W[3][9][5] = g6;
        Cm.W[3][9][6] = g7;
        Cm.W[3][10][0] = -(g8 + g9 + g10);
        Cm.W[3][10][7] = g8;
        Cm.W[3][10][8] = g9;
        Cm.W[3][10][9] = g10;
        Cm.W[3][11][0] = -(g5 + g6 + g7);
        Cm.W[3][11][4] = g5;
        Cm.W[3][11][5] = g6;
        Cm.W[3][11][6] = g7;
    }
    Some(C)
}

/*===============================================================
  EOF
  ===============================================================*/
