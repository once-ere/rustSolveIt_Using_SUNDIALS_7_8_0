//! Port of `src/arkode/arkode_sprk.c` (+ `include/arkode/arkode_sprk.h`
//! folded in, per the module-naming rule).
//!
//! The symplectic-partitioned Runge-Kutta coefficient tables used by
//! SPRKStep: a pair of stage-coefficient arrays `a` (explicit /
//! "velocity" partition) and `ahat` (diagonally-implicit / "position"
//! partition).
//!
//! Handle model: `ARKodeSPRKTable = Rc<RefCell<ARKodeSPRKTableMem>>`,
//! matching the `ARKodeButcherTable` rendering fixed by the contract
//! (§5). Cloning the `Rc` is the C pointer copy;
//! `ARKodeSPRKTable_Free` is the drop of the handle.
//!
//! `ARKODE_SPRKMethodID` is a plain `i32` alias plus a constant per
//! enumerator: the C enum carries duplicate discriminants
//! (`ARKODE_MIN_SPRK_NUM == ARKODE_SPRK_EULER_1_1 == 0`,
//! `ARKODE_MAX_SPRK_NUM == ARKODE_SPRK_SOFRONIOU_10_36`), which a Rust
//! `enum` cannot express, and `arkode_sprkstep.h`'s
//! `SPRKSTEP_DEFAULT_*` are `int` constants initialized from it.
//!
//! Coefficient fidelity: every constant is transcribed digit-for-digit
//! in the C's assignment order, no fraction simplified and no
//! sub-expression folded; every `pow` goes through
//! `sundials_math::SUNRpowerR` (never `f64::powf`) so the results are
//! bit-identical to the reference build.

use std::cell::RefCell;
use std::rc::Rc;

use sundials_core::sundials_math::{SUNRpowerR, SUNRsqrt};
use sundials_core::sundials_types::*;
use sundials_core::sundials_utils::SUNFile;

use crate::arkode_butcher::{ARKodeButcherTable, ARKodeButcherTable_Alloc, ARKodeButcherTable_Write};
use crate::arkode_impl::*;

/*===============================================================
  SPRK method identifiers (include/arkode/arkode_sprk.h)
  ===============================================================*/

pub type ARKODE_SPRKMethodID = i32;

pub const ARKODE_SPRK_NONE: ARKODE_SPRKMethodID = -1; /* ensure enum is signed int */
pub const ARKODE_SPRK_EULER_1_1: ARKODE_SPRKMethodID = 0;
pub const ARKODE_MIN_SPRK_NUM: ARKODE_SPRKMethodID = 0;
pub const ARKODE_SPRK_LEAPFROG_2_2: ARKODE_SPRKMethodID = 1;
pub const ARKODE_SPRK_PSEUDO_LEAPFROG_2_2: ARKODE_SPRKMethodID = 2;
pub const ARKODE_SPRK_RUTH_3_3: ARKODE_SPRKMethodID = 3;
pub const ARKODE_SPRK_MCLACHLAN_2_2: ARKODE_SPRKMethodID = 4;
pub const ARKODE_SPRK_MCLACHLAN_3_3: ARKODE_SPRKMethodID = 5;
pub const ARKODE_SPRK_CANDY_ROZMUS_4_4: ARKODE_SPRKMethodID = 6;
pub const ARKODE_SPRK_MCLACHLAN_4_4: ARKODE_SPRKMethodID = 7;
pub const ARKODE_SPRK_MCLACHLAN_5_6: ARKODE_SPRKMethodID = 8;
pub const ARKODE_SPRK_YOSHIDA_6_8: ARKODE_SPRKMethodID = 9;
pub const ARKODE_SPRK_SUZUKI_UMENO_8_16: ARKODE_SPRKMethodID = 10;
pub const ARKODE_SPRK_SOFRONIOU_10_36: ARKODE_SPRKMethodID = 11;
pub const ARKODE_MAX_SPRK_NUM: ARKODE_SPRKMethodID = ARKODE_SPRK_SOFRONIOU_10_36;

/*===============================================================
  SPRK table structure
  ===============================================================*/

pub struct ARKodeSPRKTableMem {
    /* method order of accuracy */
    pub q: i32,
    /* number of stages */
    pub stages: i32,
    /* the a_i coefficients generate the explicit Butcher table */
    pub a: Vec<sunrealtype>,
    /* the ahat_i coefficients generate the diagonally-implicit Butcher table */
    pub ahat: Vec<sunrealtype>,
}

pub type ARKodeSPRKTable = Rc<RefCell<ARKodeSPRKTableMem>>;

/*===============================================================
  Method tables
  ===============================================================*/

fn arkodeSymplecticEuler() -> Option<ARKodeSPRKTable> {
    let sprk_table = ARKodeSPRKTable_Alloc(1)?;
    {
        let mut t = sprk_table.borrow_mut();
        t.q = 1;
        t.stages = 1;
        t.a[0] = 1.0;
        t.ahat[0] = 1.0;
    }
    Some(sprk_table)
}

/*
  The following methods are from:

  J Candy, W Rozmus, A symplectic integration algorithm for separable
  Hamiltonian functions, Journal of Computational Physics, Volume 92, Issue 1,
  1991, Pages 230-256, ISSN 0021-9991,
  https://doi.org/10.1016/0021-9991(91)90299-Z.
 */

fn arkodeSymplecticLeapfrog2() -> Option<ARKodeSPRKTable> {
    let sprk_table = ARKodeSPRKTable_Alloc(2)?;
    {
        let mut t = sprk_table.borrow_mut();
        t.q = 2;
        t.stages = 2;
        t.a[0] = 0.5;
        t.a[1] = 0.5;
        t.ahat[0] = 0.0;
        t.ahat[1] = 1.0;
    }
    Some(sprk_table)
}

fn arkodeSymplecticPseudoLeapfrog2() -> Option<ARKodeSPRKTable> {
    let sprk_table = ARKodeSPRKTable_Alloc(2)?;
    {
        let mut t = sprk_table.borrow_mut();
        t.q = 2;
        t.stages = 2;
        t.a[0] = 1.0;
        t.a[1] = 0.0;
        t.ahat[0] = 0.5;
        t.ahat[1] = 0.5;
    }
    Some(sprk_table)
}

fn arkodeSymplecticCandyRozmus4() -> Option<ARKodeSPRKTable> {
    let sprk_table = ARKodeSPRKTable_Alloc(4)?;
    {
        let mut t = sprk_table.borrow_mut();
        t.q = 4;
        t.stages = 4;
        t.a[0] = (2.0 + SUNRpowerR(2.0, 1.0 / 3.0) + SUNRpowerR(2.0, -1.0 / 3.0)) / 6.0;
        t.a[1] = (1.0 - SUNRpowerR(2.0, 1.0 / 3.0) - SUNRpowerR(2.0, -1.0 / 3.0)) / 6.0;
        t.a[2] = t.a[1];
        t.a[3] = t.a[0];
        t.ahat[0] = 0.0;
        t.ahat[1] = 1.0 / (2.0 - SUNRpowerR(2.0, 1.0 / 3.0));
        t.ahat[2] = 1.0 / (1.0 - SUNRpowerR(2.0, 2.0 / 3.0));
        t.ahat[3] = t.ahat[1];
    }
    Some(sprk_table)
}

/*
  The following methods are from:

  Ruth, R. D. (1983). A CANONICAL INTEGRATION TECHNIQUE.
  IEEE Transactions on Nuclear Science, 30(4).
  https://accelconf.web.cern.ch/p83/PDF/PAC1983_2669.PDF
 */

fn arkodeSymplecticRuth3() -> Option<ARKodeSPRKTable> {
    let sprk_table = ARKodeSPRKTable_Alloc(3)?;
    {
        let mut t = sprk_table.borrow_mut();
        t.q = 3;
        t.stages = 3;
        t.a[0] = 2.0 / 3.0;
        t.a[1] = -2.0 / 3.0;
        t.a[2] = 1.0;
        t.ahat[0] = 7.0 / 24.0;
        t.ahat[1] = 3.0 / 4.0;
        t.ahat[2] = -1.0 / 24.0;
    }
    Some(sprk_table)
}

/*
  The following methods are from:

  McLachlan, R.I., Atela, P.: The accuracy of symplectic integrators.
  Nonlinearity. 5, 541-562 (1992). https://doi.org/10.1088/0951-7715/5/2/011
 */

fn arkodeSymplecticMcLachlan2() -> Option<ARKodeSPRKTable> {
    let sprk_table = ARKodeSPRKTable_Alloc(2)?;
    {
        let mut t = sprk_table.borrow_mut();
        t.q = 2;
        t.stages = 2;
        t.a[1] = 1.0 - (1.0 / 2.0) * SUNRsqrt(2.0);
        t.a[0] = 1.0 - t.a[1];
        t.ahat[1] = 1.0 / (2.0 * (1.0 - t.a[1]));
        t.ahat[0] = 1.0 - t.ahat[1];
    }
    Some(sprk_table)
}

fn arkodeSymplecticMcLachlan3() -> Option<ARKodeSPRKTable> {
    /* C declares w, y, z = 0.0 and overwrites them below; the initial
       values are never read. */
    let sprk_table = ARKodeSPRKTable_Alloc(3)?;

    let z: sunrealtype = -SUNRpowerR((2.0 / 27.0) - 1.0 / (9.0 * SUNRsqrt(3.0)), 1.0 / 3.0);
    let w: sunrealtype = -2.0 / 3.0 + 1.0 / (9.0 * z) + z;
    let y: sunrealtype = (1.0 + w * w) / 4.0;

    {
        let mut t = sprk_table.borrow_mut();
        t.q = 3;
        t.stages = 3;

        t.a[0] = SUNRsqrt(1.0 / (9.0 * y) - w / 2.0 + SUNRsqrt(y)) - 1.0 / (3.0 * SUNRsqrt(y));
        t.a[1] = 0.25 / t.a[0] - t.a[0] / 2.0;
        t.a[2] = 1.0 - t.a[0] - t.a[1];
        t.ahat[0] = t.a[2];
        t.ahat[1] = t.a[1];
        t.ahat[2] = t.a[0];
    }
    Some(sprk_table)
}

fn arkodeSymplecticMcLachlan4() -> Option<ARKodeSPRKTable> {
    let sprk_table = ARKodeSPRKTable_Alloc(4)?;
    {
        let mut t = sprk_table.borrow_mut();
        t.q = 4;
        t.stages = 4;
        t.a[0] = 0.515352837431122936;
        t.a[1] = -0.085782019412973646;
        t.a[2] = 0.441583023616466524;
        t.a[3] = 0.128846158365384185;
        t.ahat[0] = 0.134496199277431089;
        t.ahat[1] = -0.224819803079420806;
        t.ahat[2] = 0.756320000515668291;
        t.ahat[3] = 0.33400360328632142;
    }
    Some(sprk_table)
}

fn arkodeSymplecticMcLachlan5() -> Option<ARKodeSPRKTable> {
    let sprk_table = ARKodeSPRKTable_Alloc(6)?;
    {
        let mut t = sprk_table.borrow_mut();
        t.q = 5;
        t.stages = 6;
        t.a[0] = 0.339839625839110000;
        t.a[1] = -0.088601336903027329;
        t.a[2] = 0.5858564768259621188;
        t.a[3] = -0.603039356536491888;
        t.a[4] = 0.3235807965546976394;
        t.a[5] = 0.4423637942197494587;
        t.ahat[0] = 0.1193900292875672758;
        t.ahat[1] = 0.6989273703824752308;
        t.ahat[2] = -0.1713123582716007754;
        t.ahat[3] = 0.4012695022513534480;
        t.ahat[4] = 0.0107050818482359840;
        t.ahat[5] = -0.0589796254980311632;
    }
    Some(sprk_table)
}

/*
  The following methods are from:

  Yoshida, H.: Construction of higher order symplectic integrators.
  Phys Lett A. 150, 262-268 (1990).
  https://doi.org/10.1016/0375-9601(90)90092-3

 */

fn arkodeSymplecticYoshida6() -> Option<ARKodeSPRKTable> {
    let sprk_table = ARKodeSPRKTable_Alloc(8)?;
    {
        let mut t = sprk_table.borrow_mut();
        t.q = 6;
        t.stages = 8;
        t.a[0] = 0.7845136104775572638194976338663498757768;
        t.a[1] = 0.2355732133593581336847931829785346016865;
        t.a[2] = -1.177679984178871006946415680964315734639;
        t.a[3] = 1.315186320683911218884249728238862514352;
        t.a[4] = t.a[2];
        t.a[5] = t.a[1];
        t.a[6] = t.a[0];
        t.a[7] = 0.0;
        t.ahat[0] = t.a[0] / 2.0;
        t.ahat[1] = (t.a[0] + t.a[1]) / 2.0;
        t.ahat[2] = (t.a[1] + t.a[2]) / 2.0;
        t.ahat[3] = (t.a[2] + t.a[3]) / 2.0;
        t.ahat[4] = t.ahat[3];
        t.ahat[5] = t.ahat[2];
        t.ahat[6] = t.ahat[1];
        t.ahat[7] = t.ahat[0];
    }
    Some(sprk_table)
}

/*
  The following methods are from:

  (Original) Suzuki, M., & Umeno, K. (1993). Higher-order decomposition theory
  of exponential operators and its applications to QMC and nonlinear dynamics.
  Computer simulation studies in condensed-matter physics VI, 74-86.
  https://doi.org/10.1007/978-3-642-78448-4_7

  McLachlan, R.I.: On the Numerical Integration of Ordinary Differential
  Equations by Symmetric Composition Methods. Siam J Sci Comput. 16, 151-168
  (1995). https://doi.org/10.1137/0916010

 */

fn arkodeSymplecticSuzukiUmeno816() -> Option<ARKodeSPRKTable> {
    let sprk_table = ARKodeSPRKTable_Alloc(16)?;
    {
        let mut t = sprk_table.borrow_mut();
        t.q = 8;
        t.stages = 16;
        t.a[0] = 0.7416703643506129534482278017838063156035;
        t.a[1] = -0.4091008258000315939973000958935634173099;
        t.a[2] = 0.1907547102962383799538762564503716627355;
        t.a[3] = -0.5738624711160822666563877266355357421595;
        t.a[4] = 0.2990641813036559238444635406886029882258;
        t.a[5] = 0.3346249182452981837849579798821822886337;
        t.a[6] = 0.3152930923967665966320566638110024309941;
        t.a[7] = -0.7968879393529163540197888401737330534463;
        t.a[8] = t.a[6];
        t.a[9] = t.a[5];
        t.a[10] = t.a[4];
        t.a[11] = t.a[3];
        t.a[12] = t.a[2];
        t.a[13] = t.a[1];
        t.a[14] = t.a[0];
        t.a[15] = 0.0;
        t.ahat[0] = t.a[0] / 2.0;
        t.ahat[1] = (t.a[0] + t.a[1]) / 2.0;
        t.ahat[2] = (t.a[1] + t.a[2]) / 2.0;
        t.ahat[3] = (t.a[2] + t.a[3]) / 2.0;
        t.ahat[4] = (t.a[3] + t.a[4]) / 2.0;
        t.ahat[5] = (t.a[4] + t.a[5]) / 2.0;
        t.ahat[6] = (t.a[5] + t.a[6]) / 2.0;
        t.ahat[7] = (t.a[6] + t.a[7]) / 2.0;
        t.ahat[8] = t.ahat[7];
        t.ahat[9] = t.ahat[6];
        t.ahat[10] = t.ahat[5];
        t.ahat[11] = t.ahat[4];
        t.ahat[12] = t.ahat[3];
        t.ahat[13] = t.ahat[2];
        t.ahat[14] = t.ahat[1];
        t.ahat[15] = t.ahat[0];
    }
    Some(sprk_table)
}

/*
  The following methods are from:

  Sofroniou, M., Spaletta, G.: Derivation of symmetric composition constants for
  symmetric integrators. Optim Methods Softw. 20, 597-613 (2005).
  https://doi.org/10.1080/10556780500140664

 */

fn arkodeSymplecticSofroniou10() -> Option<ARKodeSPRKTable> {
    let sprk_table = ARKodeSPRKTable_Alloc(36)?;
    {
        let mut t = sprk_table.borrow_mut();
        t.q = 10;
        t.stages = 36;

        t.a[0] = 0.078795722521686419263907679337684;
        t.a[1] = 0.31309610341510852776481247192647;
        t.a[2] = 0.027918383235078066109520273275299;
        t.a[3] = -0.22959284159390709415121339679655;
        t.a[4] = 0.13096206107716486317465685927961;
        t.a[5] = -0.26973340565451071434460973222411;
        t.a[6] = 0.074973343155891435666137105641410;
        t.a[7] = 0.11199342399981020488957508073640;
        t.a[8] = 0.36613344954622675119314812353150;
        t.a[9] = -0.39910563013603589787862981058340;
        t.a[10] = 0.10308739852747107731580277001372;
        t.a[11] = 0.41143087395589023782070411897608;
        t.a[12] = -0.0048663605831352617621956593099771;
        t.a[13] = -0.39203335370863990644808193642610;
        t.a[14] = 0.051942502962449647037182904015976;
        t.a[15] = 0.050665090759924496335874344156866;
        t.a[16] = 0.049674370639729879054568800279461;
        t.a[17] = 0.049317735759594537917680008339338;
        t.a[18] = t.a[16];
        t.a[19] = t.a[15];
        t.a[20] = t.a[14];
        t.a[21] = t.a[13];
        t.a[22] = t.a[12];
        t.a[23] = t.a[11];
        t.a[24] = t.a[10];
        t.a[25] = t.a[9];
        t.a[26] = t.a[8];
        t.a[27] = t.a[7];
        t.a[28] = t.a[6];
        t.a[29] = t.a[5];
        t.a[30] = t.a[4];
        t.a[31] = t.a[3];
        t.a[32] = t.a[2];
        t.a[33] = t.a[1];
        t.a[34] = t.a[0];
        t.a[35] = 0.0;
        t.ahat[0] = t.a[0] / 2.0;
        t.ahat[1] = (t.a[0] + t.a[1]) / 2.0;
        t.ahat[2] = (t.a[1] + t.a[2]) / 2.0;
        t.ahat[3] = (t.a[2] + t.a[3]) / 2.0;
        t.ahat[4] = (t.a[3] + t.a[4]) / 2.0;
        t.ahat[5] = (t.a[4] + t.a[5]) / 2.0;
        t.ahat[6] = (t.a[5] + t.a[6]) / 2.0;
        t.ahat[7] = (t.a[6] + t.a[7]) / 2.0;
        t.ahat[8] = (t.a[7] + t.a[8]) / 2.0;
        t.ahat[9] = (t.a[8] + t.a[9]) / 2.0;
        t.ahat[10] = (t.a[9] + t.a[10]) / 2.0;
        t.ahat[11] = (t.a[10] + t.a[11]) / 2.0;
        t.ahat[12] = (t.a[11] + t.a[12]) / 2.0;
        t.ahat[13] = (t.a[12] + t.a[13]) / 2.0;
        t.ahat[14] = (t.a[13] + t.a[14]) / 2.0;
        t.ahat[15] = (t.a[14] + t.a[15]) / 2.0;
        t.ahat[16] = (t.a[15] + t.a[16]) / 2.0;
        t.ahat[17] = (t.a[16] + t.a[17]) / 2.0;
        t.ahat[18] = t.ahat[17];
        t.ahat[19] = t.ahat[16];
        t.ahat[20] = t.ahat[15];
        t.ahat[21] = t.ahat[14];
        t.ahat[22] = t.ahat[13];
        t.ahat[23] = t.ahat[12];
        t.ahat[24] = t.ahat[11];
        t.ahat[25] = t.ahat[10];
        t.ahat[26] = t.ahat[9];
        t.ahat[27] = t.ahat[8];
        t.ahat[28] = t.ahat[7];
        t.ahat[29] = t.ahat[6];
        t.ahat[30] = t.ahat[5];
        t.ahat[31] = t.ahat[4];
        t.ahat[32] = t.ahat[3];
        t.ahat[33] = t.ahat[2];
        t.ahat[34] = t.ahat[1];
        t.ahat[35] = t.ahat[0];
    }
    Some(sprk_table)
}

/*===============================================================
  Utility routines
  ===============================================================*/

pub fn ARKodeSPRKTable_Create(
    s: i32,
    q: i32,
    a_1d: &[sunrealtype],
    ahat_1d: &[sunrealtype],
) -> Option<ARKodeSPRKTable> {
    /* NULL `a` / `ahat` checks: handled by the type system */
    if s < 1 {
        return None;
    }

    let sprk_table = ARKodeSPRKTable_Alloc(s)?;

    {
        let mut t = sprk_table.borrow_mut();
        t.stages = s;
        t.q = q;

        for i in 0..s as usize {
            t.a[i] = a_1d[i];
            t.ahat[i] = ahat_1d[i];
        }
    }

    Some(sprk_table)
}

pub fn ARKodeSPRKTable_Alloc(stages: i32) -> Option<ARKodeSPRKTable> {
    /* C `malloc(stages * sizeof(sunrealtype))` with a negative `stages`
       requests an enormous block and fails -> NULL. */
    if stages < 0 {
        return None;
    }

    /* C: malloc + memset(0) of the record, then two `malloc`s for `ahat`
       and `a` (left uninitialized), then `sprk_table->stages = stages`.
       Every constructor above and `_Create`/`_Copy` write all `stages`
       entries of both arrays before any read, so the zero fill here is
       unobservable. */
    Some(Rc::new(RefCell::new(ARKodeSPRKTableMem {
        q: 0,
        stages,
        a: vec![0.0; stages as usize],
        ahat: vec![0.0; stages as usize],
    })))
}

pub fn ARKodeSPRKTable_Load(id: ARKODE_SPRKMethodID) -> Option<ARKodeSPRKTable> {
    match id {
        ARKODE_SPRK_EULER_1_1 => arkodeSymplecticEuler(),
        ARKODE_SPRK_LEAPFROG_2_2 => arkodeSymplecticLeapfrog2(),
        ARKODE_SPRK_PSEUDO_LEAPFROG_2_2 => arkodeSymplecticPseudoLeapfrog2(),
        ARKODE_SPRK_RUTH_3_3 => arkodeSymplecticRuth3(),
        ARKODE_SPRK_MCLACHLAN_2_2 => arkodeSymplecticMcLachlan2(),
        ARKODE_SPRK_MCLACHLAN_3_3 => arkodeSymplecticMcLachlan3(),
        ARKODE_SPRK_MCLACHLAN_4_4 => arkodeSymplecticMcLachlan4(),
        ARKODE_SPRK_CANDY_ROZMUS_4_4 => arkodeSymplecticCandyRozmus4(),
        ARKODE_SPRK_MCLACHLAN_5_6 => arkodeSymplecticMcLachlan5(),
        ARKODE_SPRK_YOSHIDA_6_8 => arkodeSymplecticYoshida6(),
        ARKODE_SPRK_SUZUKI_UMENO_8_16 => arkodeSymplecticSuzukiUmeno816(),
        ARKODE_SPRK_SOFRONIOU_10_36 => arkodeSymplecticSofroniou10(),
        _ => None,
    }
}

pub fn ARKodeSPRKTable_LoadByName(method: &str) -> Option<ARKodeSPRKTable> {
    if method == "ARKODE_SPRK_EULER_1_1" {
        return arkodeSymplecticEuler();
    }
    if method == "ARKODE_SPRK_LEAPFROG_2_2" {
        return arkodeSymplecticLeapfrog2();
    }
    if method == "ARKODE_SPRK_PSEUDO_LEAPFROG_2_2" {
        return arkodeSymplecticPseudoLeapfrog2();
    }
    if method == "ARKODE_SPRK_RUTH_3_3" {
        return arkodeSymplecticRuth3();
    }
    if method == "ARKODE_SPRK_MCLACHLAN_2_2" {
        return arkodeSymplecticMcLachlan2();
    }
    if method == "ARKODE_SPRK_MCLACHLAN_3_3" {
        return arkodeSymplecticMcLachlan3();
    }
    if method == "ARKODE_SPRK_MCLACHLAN_4_4" {
        return arkodeSymplecticMcLachlan4();
    }
    if method == "ARKODE_SPRK_CANDY_ROZMUS_4_4" {
        return arkodeSymplecticCandyRozmus4();
    }
    if method == "ARKODE_SPRK_MCLACHLAN_5_6" {
        return arkodeSymplecticMcLachlan5();
    }
    if method == "ARKODE_SPRK_YOSHIDA_6_8" {
        return arkodeSymplecticYoshida6();
    }
    if method == "ARKODE_SPRK_SUZUKI_UMENO_8_16" {
        return arkodeSymplecticSuzukiUmeno816();
    }
    if method == "ARKODE_SPRK_SOFRONIOU_10_36" {
        return arkodeSymplecticSofroniou10();
    }
    None
}

pub fn ARKodeSPRKTable_Copy(that_sprk_table: &ARKodeSPRKTable) -> Option<ARKodeSPRKTable> {
    let sprk_table = ARKodeSPRKTable_Alloc(that_sprk_table.borrow().stages);

    /* C dereferences the result without a NULL check (UB on allocation
       failure) -> deterministic panic. */
    let sprk_table = sprk_table.expect("ARKodeSPRKTable_Alloc");

    {
        let that = that_sprk_table.borrow();
        let mut t = sprk_table.borrow_mut();

        t.q = that.q;

        /* C loop bound `sprk_table->stages`, copied out so the `for` head
           holds no borrow of the guard across the body. */
        let stages = t.stages;
        for i in 0..stages as usize {
            t.ahat[i] = that.ahat[i];
            t.a[i] = that.a[i];
        }
    }

    Some(sprk_table)
}

pub fn ARKodeSPRKTable_Space(
    sprk_table: &ARKodeSPRKTable,
    liw: &mut sunindextype,
    lrw: &mut sunindextype,
) {
    *liw = 2;
    *lrw = (sprk_table.borrow().stages * 2) as sunindextype;
}

/// C `ARKodeSPRKTable_Free(sprk_table)`: frees `ahat`, `a`, then the
/// record. Here the handle is taken by value and dropped; the last
/// outstanding `Rc` releases all three. `None` is C's NULL (a no-op).
pub fn ARKodeSPRKTable_Free(sprk_table: Option<ARKodeSPRKTable>) {
    drop(sprk_table);
}

pub fn ARKodeSPRKTable_Write(sprk_table: &ARKodeSPRKTable, outfile: &SUNFile) {
    let mut a: Option<ARKodeButcherTable> = None;
    let mut b: Option<ARKodeButcherTable> = None;

    let _ = ARKodeSPRKTable_ToButcher(sprk_table, &mut a, &mut b);

    /* C passes the (possibly NULL) tables straight through and
       ARKodeButcherTable_Write returns silently on NULL; the port forwards
       the `Option` unchanged so a failed `_ToButcher` prints nothing here
       exactly as in C. */
    ARKodeButcherTable_Write(a.as_ref(), outfile);
    ARKodeButcherTable_Write(b.as_ref(), outfile);

    /* C `ARKodeButcherTable_Free(a); ARKodeButcherTable_Free(b);` -- the
       frozen contract (§5) renders the Butcher `_Free` as `drop`. */
    drop(a);
    drop(b);
}

/// C `ARKodeSPRKTable_ToButcher`.
///
/// NOTE: upstream's loop nest reuses the *outer* index `i` inside the
/// two "time weights" loops and inside the explicit-table loop, so the
/// outer `i` loop always terminates after a single pass (for every
/// `stages >= 1`) and only row 0 of the implicit table's `A`/`b` is
/// written. That is transcribed literally here -- index variables are
/// explicit `while` loops sharing `i`/`j` exactly as in C.
pub fn ARKodeSPRKTable_ToButcher(
    sprk_table: &ARKodeSPRKTable,
    a_ptr: &mut Option<ARKodeButcherTable>,
    b_ptr: &mut Option<ARKodeButcherTable>,
) -> i32 {
    let mut i: i32 = 0;
    /* C: `int j = 0;` -- the initializer is never read (every loop below
       re-arms `j`), so it is left off to keep the build warning-free. */
    let mut j: i32;

    let stages = sprk_table.borrow().stages;

    let a = match ARKodeButcherTable_Alloc(stages, SUNFALSE) {
        Some(a) => a,
        None => return ARK_MEM_FAIL,
    };
    let b = match ARKodeButcherTable_Alloc(stages, SUNFALSE) {
        Some(b) => b,
        None => {
            /* C: `if (a) { ARKodeButcherTable_Free(a); }` */
            drop(a);
            return ARK_MEM_FAIL;
        }
    };

    {
        let sprk = sprk_table.borrow();
        let mut at = a.borrow_mut();
        let mut bt = b.borrow_mut();

        /* DIRK table */
        while i < stages {
            bt.b[i as usize] = sprk.ahat[i as usize];
            j = 0;
            while j <= i {
                bt.A[i as usize][j as usize] = sprk.ahat[j as usize];
                j += 1;
            }
            /* Time weights: C_j = sum_{i=0}^{j} b_i */

            /* Time weights: C_j = sum_{i=0}^{j-1} b_i */
            j = 0;
            while j < stages {
                i = 0;
                while i <= j {
                    bt.c[j as usize] += sprk.ahat[i as usize];
                    i += 1;
                }
                j += 1;
            }

            /* Explicit table */
            i = 0;
            while i < stages {
                at.b[i as usize] = sprk.a[i as usize];
                j = 0;
                while j < i {
                    at.A[i as usize][j as usize] = sprk.a[j as usize];
                    j += 1;
                }
                i += 1;
            }

            /* Time weights: c_j = sum_{i=0}^{j-1} a_i */
            j = 0;
            while j < stages {
                i = 0;
                while i < j {
                    at.c[j as usize] += sprk.a[i as usize];
                    i += 1;
                }
                j += 1;
            }

            /* Set method order */
            at.q = sprk.q;
            bt.q = sprk.q;

            /* No embedding, so set embedding order to 0 */
            at.p = 0;
            bt.p = 0;

            i += 1;
        }
    }

    *a_ptr = Some(a);
    *b_ptr = Some(b);

    ARK_SUCCESS
}
