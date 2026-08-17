//! Port of `src/arkode/arkode_butcher_erk.c` +
//! `include/arkode/arkode_butcher_erk.h`, with the X-macro coefficient
//! table `src/arkode/arkode_butcher_erk.def` expanded in place exactly
//! as the C preprocessor expands it into the three `ARK_BUTCHER_TABLE`
//! sites: the `switch` in `ARKodeButcherTable_LoadERK`, the `switch` in
//! `ARKodeButcherTable_ERKIDToName`, and the `strcmp` chain in
//! `arkButcherTableERKNameToID`.
//!
//! The tables are pure DATA: every coefficient is transcribed digit for
//! digit in the C's own order, no fraction is simplified and no constant
//! is folded. `SUN_RCONST(x)` is a bare `f64` literal in Rust, and a
//! written-out quotient such as `1.0 / 3.0` stays a runtime IEEE-754
//! division, bit-identical to the C compiler's constant folding.
//!
//! `enum ARKODE_ERKTableID` becomes an `i32` alias plus consts carrying the
//! exact C values, not a Rust `enum`, for three reasons that all bear on
//! fidelity:
//!  * the C enum has two value ALIASES (`ARKODE_MIN_ERK_NUM == ARKODE_HEUN_EULER_2_1_2`,
//!    `ARKODE_MAX_ERK_NUM == ARKODE_ASCHER_ERK_3_1_2`) and Rust forbids duplicate
//!    discriminants;
//!  * callers compare IDs with `<`, `>` and `> -1`
//!    (`arkode_erkstep_io.c`, `arkode_arkstep.c`);
//!  * reference examples pass a bare `-1` for "no table"
//!    (`ark_brusselator_mri.c`, `ark_reaction_diffusion_mri.c`), which
//!    must keep compiling verbatim.

use crate::arkode_butcher::{ARKodeButcherTable, ARKodeButcherTable_Alloc};
use crate::arkode_impl::*;
use sundials_core::sundials_types::*;

/* =================================================================
 * enum ARKODE_ERKTableID (include/arkode/arkode_butcher_erk.h)
 * =================================================================*/

pub type ARKODE_ERKTableID = i32;

pub const ARKODE_ERK_NONE: ARKODE_ERKTableID = -1; /* ensure enum is signed int */
pub const ARKODE_HEUN_EULER_2_1_2: ARKODE_ERKTableID = 0;
pub const ARKODE_MIN_ERK_NUM: ARKODE_ERKTableID = 0;
pub const ARKODE_BOGACKI_SHAMPINE_4_2_3: ARKODE_ERKTableID = 1;
pub const ARKODE_ARK324L2SA_ERK_4_2_3: ARKODE_ERKTableID = 2;
pub const ARKODE_ZONNEVELD_5_3_4: ARKODE_ERKTableID = 3;
pub const ARKODE_ARK436L2SA_ERK_6_3_4: ARKODE_ERKTableID = 4;
pub const ARKODE_SAYFY_ABURUB_6_3_4: ARKODE_ERKTableID = 5;
pub const ARKODE_CASH_KARP_6_4_5: ARKODE_ERKTableID = 6;
pub const ARKODE_FEHLBERG_6_4_5: ARKODE_ERKTableID = 7;
pub const ARKODE_DORMAND_PRINCE_7_4_5: ARKODE_ERKTableID = 8;
pub const ARKODE_ARK548L2SA_ERK_8_4_5: ARKODE_ERKTableID = 9;
pub const ARKODE_VERNER_8_5_6: ARKODE_ERKTableID = 10;
pub const ARKODE_FEHLBERG_13_7_8: ARKODE_ERKTableID = 11;
pub const ARKODE_KNOTH_WOLKE_3_3: ARKODE_ERKTableID = 12;
pub const ARKODE_ARK437L2SA_ERK_7_3_4: ARKODE_ERKTableID = 13;
pub const ARKODE_ARK548L2SAb_ERK_8_4_5: ARKODE_ERKTableID = 14;
pub const ARKODE_ARK2_ERK_3_1_2: ARKODE_ERKTableID = 15;
pub const ARKODE_SOFRONIOU_SPALETTA_5_3_4: ARKODE_ERKTableID = 16;
pub const ARKODE_SHU_OSHER_3_2_3: ARKODE_ERKTableID = 17;
pub const ARKODE_VERNER_9_5_6: ARKODE_ERKTableID = 18;
pub const ARKODE_VERNER_10_6_7: ARKODE_ERKTableID = 19;
pub const ARKODE_VERNER_13_7_8: ARKODE_ERKTableID = 20;
pub const ARKODE_VERNER_16_8_9: ARKODE_ERKTableID = 21;
pub const ARKODE_FORWARD_EULER_1_1: ARKODE_ERKTableID = 22;
pub const ARKODE_RALSTON_EULER_2_1_2: ARKODE_ERKTableID = 23;
pub const ARKODE_EXPLICIT_MIDPOINT_EULER_2_1_2: ARKODE_ERKTableID = 24;
pub const ARKODE_RALSTON_3_1_2: ARKODE_ERKTableID = 25;
pub const ARKODE_TSITOURAS_7_4_5: ARKODE_ERKTableID = 26;
pub const ARKODE_SSP_ERK_3_1_2: ARKODE_ERKTableID = 27;
pub const ARKODE_SSP_ERK_4_1_2: ARKODE_ERKTableID = 28;
pub const ARKODE_SSP_ERK_4_2_3: ARKODE_ERKTableID = 29;
pub const ARKODE_SSP_ERK_10_3_4: ARKODE_ERKTableID = 30;
pub const ARKODE_SSP_LSPUM_ERK_3_1_2: ARKODE_ERKTableID = 31;
pub const ARKODE_ASCHER_ERK_3_1_2: ARKODE_ERKTableID = 32;
pub const ARKODE_MAX_ERK_NUM: ARKODE_ERKTableID = 32;

/*---------------------------------------------------------------
  Returns Butcher table structure for pre-set Runge Kutta methods.

  Input:  emethod -- integer key for the desired method
  ---------------------------------------------------------------*/
pub fn ARKodeButcherTable_LoadERK(emethod: ARKODE_ERKTableID) -> Option<ARKodeButcherTable> {
    /* Use X-macro to test each method name */
    match emethod {
        ARKODE_ERK_NONE => None,
        ARKODE_FORWARD_EULER_1_1 => {
            /* Euler-ERK */
            let B = ARKodeButcherTable_Alloc(1, SUNFALSE)?;
            {
                let mut Bm = B.borrow_mut();
                let B = &mut *Bm;

                B.q = 1;
                B.p = 0;

                B.b[0] = 1.0;
            }
            Some(B)
        }
        ARKODE_RALSTON_3_1_2 => {
            /* Ralston-ERK */
            let B = ARKodeButcherTable_Alloc(3, SUNTRUE)?;
            {
                let mut Bm = B.borrow_mut();
                let B = &mut *Bm;

                B.q = 2;
                B.p = 1;

                B.A[1][0] = 2.0 / 3.0;
                B.A[2][0] = 1.0 / 4.0;
                B.A[2][1] = 3.0 / 4.0;

                B.b[0] = 1.0 / 4.0;
                B.b[1] = 3.0 / 4.0;
                B.b[2] = 0.0;

                B.d[0] = 5.0 / 37.0;
                B.d[1] = 2.0 / 3.0;
                B.d[2] = 22.0 / 111.0;

                B.c[1] = 2.0 / 3.0;
                B.c[2] = 1.0;
            }
            Some(B)
        }
        ARKODE_HEUN_EULER_2_1_2 => {
            /* Heun-Euler-ERK */
            let B = ARKodeButcherTable_Alloc(2, SUNTRUE)?;
            {
                let mut Bm = B.borrow_mut();
                let B = &mut *Bm;

                B.q = 2;
                B.p = 1;

                B.A[1][0] = 1.0;

                B.b[0] = 1.0 / 2.0;
                B.b[1] = 1.0 / 2.0;

                B.d[0] = 1.0;

                B.c[1] = 1.0;
            }
            Some(B)
        }
        ARKODE_RALSTON_EULER_2_1_2 => {
            /* Ralston-Euler-ERK */
            let B = ARKodeButcherTable_Alloc(2, SUNTRUE)?;
            {
                let mut Bm = B.borrow_mut();
                let B = &mut *Bm;

                B.q = 2;
                B.p = 1;

                B.A[1][0] = 2.0 / 3.0;

                B.b[0] = 1.0 / 4.0;
                B.b[1] = 3.0 / 4.0;

                B.d[0] = 1.0;

                B.c[1] = 2.0 / 3.0;
            }
            Some(B)
        }
        ARKODE_EXPLICIT_MIDPOINT_EULER_2_1_2 => {
            /* Explicit-Midpoint-Euler-ERK */
            let B = ARKodeButcherTable_Alloc(2, SUNTRUE)?;
            {
                let mut Bm = B.borrow_mut();
                let B = &mut *Bm;

                B.q = 2;
                B.p = 1;

                B.A[1][0] = 1.0 / 2.0;

                B.b[1] = 1.0;

                B.d[0] = 1.0;

                B.c[1] = 1.0 / 2.0;
            }
            Some(B)
        }
        ARKODE_ARK2_ERK_3_1_2 => {
            /* ARK2 Explicit Table */
            let B = ARKodeButcherTable_Alloc(3, SUNTRUE)?;
            {
                let mut Bm = B.borrow_mut();
                let B = &mut *Bm;

                /* 1 - 1 / sqrt(2) */
                let gamma: sunrealtype = 0.2928932188134524755991556378951509607152;
                /* (3 + 2 sqrt(2)) / 6 */
                let alpha: sunrealtype = 0.9714045207910316829338962414032326928566;
                /* 1 / (2 sqrt(2)) */
                let delta: sunrealtype = 0.3535533905932737622004221810524245196424;
                /* 2 - sqrt(2) */
                let twogamma: sunrealtype = 2.0 * gamma;
                /* (4 - sqrt(2)) / 8 */
                let beta: sunrealtype = 0.3232233047033631188997889094737877401788;

                B.q = 2;
                B.p = 1;

                B.A[1][0] = twogamma;
                B.A[2][0] = 1.0 - alpha;
                B.A[2][1] = alpha;

                B.b[0] = delta;
                B.b[1] = delta;
                B.b[2] = gamma;

                B.d[0] = beta;
                B.d[1] = beta;
                B.d[2] = delta;

                B.c[1] = twogamma;
                B.c[2] = 1.0;
            }
            Some(B)
        }
        ARKODE_BOGACKI_SHAMPINE_4_2_3 => {
            /* Bogacki-Shampine-ERK */
            let B = ARKodeButcherTable_Alloc(4, SUNTRUE)?;
            {
                let mut Bm = B.borrow_mut();
                let B = &mut *Bm;

                B.q = 3;
                B.p = 2;
                B.A[1][0] = 1.0 / 2.0;
                B.A[2][1] = 3.0 / 4.0;
                B.A[3][0] = 2.0 / 9.0;
                B.A[3][1] = 1.0 / 3.0;
                B.A[3][2] = 4.0 / 9.0;

                B.b[0] = 2.0 / 9.0;
                B.b[1] = 1.0 / 3.0;
                B.b[2] = 4.0 / 9.0;

                B.d[0] = 7.0 / 24.0;
                B.d[1] = 1.0 / 4.0;
                B.d[2] = 1.0 / 3.0;
                B.d[3] = 1.0 / 8.0;

                B.c[1] = 1.0 / 2.0;
                B.c[2] = 3.0 / 4.0;
                B.c[3] = 1.0;
            }
            Some(B)
        }
        ARKODE_SSP_ERK_3_1_2 => {
            /* https://doi.org/10.1016/j.cam.2022.114325 */
            let B = ARKodeButcherTable_Alloc(3, SUNTRUE)?;
            {
                let mut Bm = B.borrow_mut();
                let B = &mut *Bm;

                B.q = 2;
                B.p = 1;
                B.A[1][0] = 0.5;
                B.A[2][0] = 0.5;
                B.A[2][1] = 0.5;

                B.b[0] = 1.0 / 3.0;
                B.b[1] = 1.0 / 3.0;
                B.b[2] = 1.0 / 3.0;

                B.d[0] = 4.0 / 9.0;
                B.d[1] = 1.0 / 3.0;
                B.d[2] = 2.0 / 9.0;

                B.c[1] = 0.5;
                B.c[2] = 1.0;
            }
            Some(B)
        }
        ARKODE_SSP_ERK_4_1_2 => {
            /* TODO(DRR): Add citation */
            let B = ARKodeButcherTable_Alloc(4, SUNTRUE)?;
            {
                let mut Bm = B.borrow_mut();
                let B = &mut *Bm;

                /* 1/3 */
                let third: sunrealtype = 1.0 / 3.0;

                B.q = 2;
                B.p = 1;
                B.A[1][0] = third;
                B.A[2][0] = third;
                B.A[2][1] = third;
                B.A[3][0] = third;
                B.A[3][1] = third;
                B.A[3][2] = third;

                B.b[0] = 0.25;
                B.b[1] = 0.25;
                B.b[2] = 0.25;
                B.b[3] = 0.25;

                B.d[0] = 5.0 / 16.0;
                B.d[1] = 1.0 / 4.0;
                B.d[2] = 1.0 / 4.0;
                B.d[3] = 3.0 / 16.0;

                B.c[1] = third;
                B.c[2] = 2.0 * third;
                B.c[3] = 1.0;
            }
            Some(B)
        }
        ARKODE_ASCHER_ERK_3_1_2 => {
            /* https://doi.org/10.1016/S0168-9274(97)00056-1 (custom embedding) */
            let B = ARKodeButcherTable_Alloc(3, SUNTRUE)?;
            {
                let mut Bm = B.borrow_mut();
                let B = &mut *Bm;

                let gamma: sunrealtype = 0.2928932188134524755991556378951509607152;
                let delta: sunrealtype = -0.7071067811865475244008443621048490392848;
                B.q = 2;
                B.p = 1;
                B.A[1][0] = gamma;
                B.A[2][0] = delta;
                B.A[2][1] = 1.707106781186547524400844362104849039285;

                B.b[0] = delta;
                B.b[1] = 1.707106781186547524400844362104849039285;

                B.d[1] = 3.0 / 5.0;
                B.d[2] = 2.0 / 5.0;

                B.c[1] = gamma;
                B.c[2] = 1.0;
            }
            Some(B)
        }
        ARKODE_SSP_ERK_4_2_3 => {
            /* https://doi.org/10.1016/j.cam.2022.114325 */
            let B = ARKodeButcherTable_Alloc(4, SUNTRUE)?;
            {
                let mut Bm = B.borrow_mut();
                let B = &mut *Bm;

                B.q = 3;
                B.p = 2;
                B.A[1][0] = 0.5;
                B.A[2][0] = 0.5;
                B.A[2][1] = 0.5;
                B.A[3][0] = 1.0 / 6.0;
                B.A[3][1] = 1.0 / 6.0;
                B.A[3][2] = 1.0 / 6.0;

                B.b[0] = 1.0 / 6.0;
                B.b[1] = 1.0 / 6.0;
                B.b[2] = 1.0 / 6.0;
                B.b[3] = 0.5;

                B.d[0] = 0.25;
                B.d[1] = 0.25;
                B.d[2] = 0.25;
                B.d[3] = 0.25;

                B.c[1] = 0.5;
                B.c[2] = 1.0;
                B.c[3] = 0.5;
            }
            Some(B)
        }
        ARKODE_SSP_ERK_10_3_4 => {
            /* https://doi.org/10.1016/j.cam.2022.114325 */
            let B = ARKodeButcherTable_Alloc(10, SUNTRUE)?;
            {
                let mut Bm = B.borrow_mut();
                let B = &mut *Bm;

                /* 1/6, 1/15, 1/10 */
                let t1: sunrealtype = 1.0 / 6.0;
                let t2: sunrealtype = 1.0 / 15.0;
                let t3: sunrealtype = 0.1;

                B.q = 4;
                B.p = 3;
                B.A[1][0] = t1;
                B.A[2][0] = t1;
                B.A[2][1] = t1;
                B.A[3][0] = t1;
                B.A[3][1] = t1;
                B.A[3][2] = t1;
                B.A[4][0] = t1;
                B.A[4][1] = t1;
                B.A[4][2] = t1;
                B.A[4][3] = t1;
                B.A[5][0] = t2;
                B.A[5][1] = t2;
                B.A[5][2] = t2;
                B.A[5][3] = t2;
                B.A[5][4] = t2;
                B.A[6][0] = t2;
                B.A[6][1] = t2;
                B.A[6][2] = t2;
                B.A[6][3] = t2;
                B.A[6][4] = t2;
                B.A[6][5] = t1;
                B.A[7][0] = t2;
                B.A[7][1] = t2;
                B.A[7][2] = t2;
                B.A[7][3] = t2;
                B.A[7][4] = t2;
                B.A[7][5] = t1;
                B.A[7][6] = t1;
                B.A[8][0] = t2;
                B.A[8][1] = t2;
                B.A[8][2] = t2;
                B.A[8][3] = t2;
                B.A[8][4] = t2;
                B.A[8][5] = t1;
                B.A[8][6] = t1;
                B.A[8][7] = t1;
                B.A[9][0] = t2;
                B.A[9][1] = t2;
                B.A[9][2] = t2;
                B.A[9][3] = t2;
                B.A[9][4] = t2;
                B.A[9][5] = t1;
                B.A[9][6] = t1;
                B.A[9][7] = t1;
                B.A[9][8] = t1;

                B.b[0] = t3;
                B.b[1] = t3;
                B.b[2] = t3;
                B.b[3] = t3;
                B.b[4] = t3;
                B.b[5] = t3;
                B.b[6] = t3;
                B.b[7] = t3;
                B.b[8] = t3;
                B.b[9] = t3;

                B.d[0] = 0.2;
                B.d[3] = 0.3;
                B.d[6] = 0.2;
                B.d[8] = 0.3;

                B.c[1] = t1;
                B.c[2] = 2.0 * t1;
                B.c[3] = 3.0 * t1;
                B.c[4] = 4.0 * t1;
                B.c[5] = 5.0 * t2;
                B.c[6] = 0.5;
                B.c[7] = 2.0 / 3.0;
                B.c[8] = 5.0 / 6.0;
                B.c[9] = 1.0;
            }
            Some(B)
        }
        ARKODE_ARK324L2SA_ERK_4_2_3 => {
            /* ARK3(2)4L[2]SA-ERK */
            let B = ARKodeButcherTable_Alloc(4, SUNTRUE)?;
            {
                let mut Bm = B.borrow_mut();
                let B = &mut *Bm;

                B.q = 3;
                B.p = 2;
                B.A[1][0] = 0.8717330430169179988320389023871136850586;
                B.A[2][0] = 0.52758901197630041156180797140291790433;
                B.A[2][1] = 0.07241098802369958843819202859708209566999;
                B.A[3][0] = 0.3990960076760701320627260736092142797856;
                B.A[3][1] = -0.437557654613519443722846363831022571942;
                B.A[3][2] = 1.038461646937449311660120290221808292156;

                B.b[0] = 0.1876410243467238251612921441668043913795;
                B.b[1] = -0.5952974735769549480478230275858851737782;
                B.b[2] = 0.9717899277217721234705114322255239398694;
                B.b[3] = 0.4358665215084589994160194511935568425293;

                B.d[0] = 0.2147402862233891404862383406484193714659;
                B.d[1] = -0.4851622638849390928209050808398155895845;
                B.d[2] = 0.86872500252038755116621237682951240796;
                B.d[3] = 0.4016969751411624011684543633618838101586;

                B.c[1] = 0.8717330430169179988320389023871136850586;
                B.c[2] = 0.6;
                B.c[3] = 1.0;
            }
            Some(B)
        }
        ARKODE_SHU_OSHER_3_2_3 => {
            /* Shu-Osher-ERK */
            let B = ARKodeButcherTable_Alloc(3, SUNTRUE)?;
            {
                let mut Bm = B.borrow_mut();
                let B = &mut *Bm;

                B.q = 3;
                B.p = 2;

                B.A[1][0] = 1.0;
                B.A[2][0] = 1.0 / 4.0;
                B.A[2][1] = 1.0 / 4.0;

                B.b[0] = 1.0 / 6.0;
                B.b[1] = 1.0 / 6.0;
                B.b[2] = 2.0 / 3.0;

                B.d[0] = 291485418878409.0 / 1.0e15;
                B.d[1] = 291485418878409.0 / 1.0e15;
                B.d[2] = 208514581121591.0 / 5.0e14;

                B.c[1] = 1.0;
                B.c[2] = 1.0 / 2.0;
            }
            Some(B)
        }
        ARKODE_SSP_LSPUM_ERK_3_1_2 => {
            let B = ARKodeButcherTable_Alloc(3, SUNTRUE)?;
            {
                let mut Bm = B.borrow_mut();
                let B = &mut *Bm;

                /* Method from https://doi.org/10.1016/j.cam.2014.05.011 */
                /* Custom embedding */
                B.q = 2;
                B.p = 1;

                B.A[1][0] = 5.0 / 6.0;
                B.A[2][0] = 11.0 / 24.0;
                B.A[2][1] = 11.0 / 24.0;

                B.b[0] = 24.0 / 55.0;
                B.b[1] = 1.0 / 5.0;
                B.b[2] = 4.0 / 11.0;

                B.d[0] = 4247.0 / 9840.0;
                B.d[1] = 21343.0 / 108240.0;
                B.d[2] = 49.0 / 132.0;

                B.c[1] = 5.0 / 6.0;
                B.c[2] = 11.0 / 12.0;
            }
            Some(B)
        }
        ARKODE_SOFRONIOU_SPALETTA_5_3_4 => {
            /* Sofroniou-Spaletta-ERK */
            let B = ARKodeButcherTable_Alloc(5, SUNTRUE)?;
            {
                let mut Bm = B.borrow_mut();
                let B = &mut *Bm;

                B.q = 4;
                B.p = 3;
                B.A[1][0] = 2.0 / 5.0;
                B.A[2][0] = -3.0 / 20.0;
                B.A[2][1] = 3.0 / 4.0;
                B.A[3][0] = 19.0 / 44.0;
                B.A[3][1] = -15.0 / 44.0;
                B.A[3][2] = 10.0 / 11.0;
                B.A[4][0] = 11.0 / 72.0;
                B.A[4][1] = 25.0 / 72.0;
                B.A[4][2] = 25.0 / 72.0;
                B.A[4][3] = 11.0 / 72.0;

                B.b[0] = 11.0 / 72.0;
                B.b[1] = 25.0 / 72.0;
                B.b[2] = 25.0 / 72.0;
                B.b[3] = 11.0 / 72.0;

                B.d[0] = 1251515.0 / 8970912.0;
                B.d[1] = 3710105.0 / 8970912.0;
                B.d[2] = 2519695.0 / 8970912.0;
                B.d[3] = 61105.0 / 8970912.0;
                B.d[4] = 119041.0 / 747576.0;

                B.c[1] = 2.0 / 5.0;
                B.c[2] = 3.0 / 5.0;
                B.c[3] = 1.0;
                B.c[4] = 1.0;
            }
            Some(B)
        }
        ARKODE_ZONNEVELD_5_3_4 => {
            /* Zonneveld */
            let B = ARKodeButcherTable_Alloc(5, SUNTRUE)?;
            {
                let mut Bm = B.borrow_mut();
                let B = &mut *Bm;

                B.q = 4;
                B.p = 3;
                B.A[1][0] = 0.5;
                B.A[2][1] = 0.5;
                B.A[3][2] = 1.0;
                B.A[4][0] = 5.0 / 32.0;
                B.A[4][1] = 7.0 / 32.0;
                B.A[4][2] = 13.0 / 32.0;
                B.A[4][3] = -1.0 / 32.0;

                B.b[0] = 1.0 / 6.0;
                B.b[1] = 1.0 / 3.0;
                B.b[2] = 1.0 / 3.0;
                B.b[3] = 1.0 / 6.0;

                B.d[0] = -1.0 / 2.0;
                B.d[1] = 7.0 / 3.0;
                B.d[2] = 7.0 / 3.0;
                B.d[3] = 13.0 / 6.0;
                B.d[4] = -16.0 / 3.0;

                B.c[1] = 0.5;
                B.c[2] = 0.5;
                B.c[3] = 1.0;
                B.c[4] = 0.75;
            }
            Some(B)
        }
        ARKODE_ARK436L2SA_ERK_6_3_4 => {
            /* ARK4(3)6L[2]SA-ERK */
            let B = ARKodeButcherTable_Alloc(6, SUNTRUE)?;
            {
                let mut Bm = B.borrow_mut();
                let B = &mut *Bm;

                B.q = 4;
                B.p = 3;
                B.A[1][0] = 0.5;
                B.A[2][0] = 13861.0 / 62500.0;
                B.A[2][1] = 6889.0 / 62500.0;
                B.A[3][0] = -116923316275.0 / 2393684061468.0;
                B.A[3][1] = -2731218467317.0 / 15368042101831.0;
                B.A[3][2] = 9408046702089.0 / 11113171139209.0;
                B.A[4][0] = -451086348788.0 / 2902428689909.0;
                B.A[4][1] = -2682348792572.0 / 7519795681897.0;
                B.A[4][2] = 12662868775082.0 / 11960479115383.0;
                B.A[4][3] = 3355817975965.0 / 11060851509271.0;
                B.A[5][0] = 647845179188.0 / 3216320057751.0;
                B.A[5][1] = 73281519250.0 / 8382639484533.0;
                B.A[5][2] = 552539513391.0 / 3454668386233.0;
                B.A[5][3] = 3354512671639.0 / 8306763924573.0;
                B.A[5][4] = 4040.0 / 17871.0;

                B.b[0] = 82889.0 / 524892.0;
                B.b[2] = 15625.0 / 83664.0;
                B.b[3] = 69875.0 / 102672.0;
                B.b[4] = -2260.0 / 8211.0;
                B.b[5] = 1.0 / 4.0;

                B.d[0] = 4586570599.0 / 29645900160.0;
                B.d[2] = 178811875.0 / 945068544.0;
                B.d[3] = 814220225.0 / 1159782912.0;
                B.d[4] = -3700637.0 / 11593932.0;
                B.d[5] = 61727.0 / 225920.0;

                B.c[1] = 1.0 / 2.0;
                B.c[2] = 83.0 / 250.0;
                B.c[3] = 31.0 / 50.0;
                B.c[4] = 17.0 / 20.0;
                B.c[5] = 1.0;
            }
            Some(B)
        }
        ARKODE_ARK437L2SA_ERK_7_3_4 => {
            /* ARK4(3)7L[2]SA-ERK */
            let B = ARKodeButcherTable_Alloc(7, SUNTRUE)?;
            {
                let mut Bm = B.borrow_mut();
                let B = &mut *Bm;

                B.q = 4;
                B.p = 3;
                B.A[1][0] = 247.0 / 1000.0;
                B.A[2][0] = 247.0 / 4000.0;
                B.A[2][1] = 2694949928731.0 / 7487940209513.0;
                B.A[3][0] = 464650059369.0 / 8764239774964.0;
                B.A[3][1] = 878889893998.0 / 2444806327765.0;
                B.A[3][2] = -952945855348.0 / 12294611323341.0;
                B.A[4][0] = 476636172619.0 / 8159180917465.0;
                B.A[4][1] = -1271469283451.0 / 7793814740893.0;
                B.A[4][2] = -859560642026.0 / 4356155882851.0;
                B.A[4][3] = 1723805262919.0 / 4571918432560.0;
                B.A[5][0] = 6338158500785.0 / 11769362343261.0;
                B.A[5][1] = -4970555480458.0 / 10924838743837.0;
                B.A[5][2] = 3326578051521.0 / 2647936831840.0;
                B.A[5][3] = -880713585975.0 / 1841400956686.0;
                B.A[5][4] = -1428733748635.0 / 8843423958496.0;
                B.A[6][0] = 760814592956.0 / 3276306540349.0;
                B.A[6][1] = 760814592956.0 / 3276306540349.0;
                B.A[6][2] = -47223648122716.0 / 6934462133451.0;
                B.A[6][3] = 71187472546993.0 / 9669769126921.0;
                B.A[6][4] = -13330509492149.0 / 9695768672337.0;
                B.A[6][5] = 11565764226357.0 / 8513123442827.0;

                B.b[2] = 9164257142617.0 / 17756377923965.0;
                B.b[3] = -10812980402763.0 / 74029279521829.0;
                B.b[4] = 1335994250573.0 / 5691609445217.0;
                B.b[5] = 2273837961795.0 / 8368240463276.0;
                B.b[6] = 247.0 / 2000.0;

                B.d[2] = 4469248916618.0 / 8635866897933.0;
                B.d[3] = -621260224600.0 / 4094290005349.0;
                B.d[4] = 696572312987.0 / 2942599194819.0;
                B.d[5] = 1532940081127.0 / 5565293938103.0;
                B.d[6] = 2441.0 / 20000.0;

                B.c[1] = 247.0 / 1000.0;
                B.c[2] = 4276536705230.0 / 10142255878289.0;
                B.c[3] = 67.0 / 200.0;
                B.c[4] = 3.0 / 40.0;
                B.c[5] = 7.0 / 10.0;
                B.c[6] = 1.0;
            }
            Some(B)
        }
        ARKODE_SAYFY_ABURUB_6_3_4 => {
            /* Sayfy-Aburub-4-3-ERK */
            let B = ARKodeButcherTable_Alloc(6, SUNTRUE)?;
            {
                let mut Bm = B.borrow_mut();
                let B = &mut *Bm;

                B.q = 4;
                B.p = 3;
                B.A[1][0] = 1.0 / 2.0;
                B.A[2][0] = -1.0;
                B.A[2][1] = 2.0;
                B.A[3][0] = 1.0 / 6.0;
                B.A[3][1] = 2.0 / 3.0;
                B.A[3][2] = 1.0 / 6.0;
                B.A[4][0] = 0.137;
                B.A[4][1] = 0.226;
                B.A[4][2] = 0.137;
                B.A[5][0] = 0.452;
                B.A[5][1] = -0.904;
                B.A[5][2] = -0.548;
                B.A[5][4] = 2.0;

                B.b[0] = 1.0 / 6.0;
                B.b[1] = 1.0 / 3.0;
                B.b[2] = 1.0 / 12.0;
                B.b[3] = 0.0;
                B.b[4] = 1.0 / 3.0;
                B.b[5] = 1.0 / 12.0;

                B.d[0] = 1.0 / 6.0;
                B.d[1] = 2.0 / 3.0;
                B.d[2] = 1.0 / 6.0;

                B.c[1] = 1.0 / 2.0;
                B.c[2] = 1.0;
                B.c[3] = 1.0;
                B.c[4] = 1.0 / 2.0;
                B.c[5] = 1.0;
            }
            Some(B)
        }
        ARKODE_TSITOURAS_7_4_5 => {
            /* Tsitouras-ERK */
            let B = ARKodeButcherTable_Alloc(7, SUNTRUE)?;
            {
                let mut Bm = B.borrow_mut();
                let B = &mut *Bm;

                B.q = 5;
                B.p = 4;
                B.A[1][0] = 0.161;
                B.A[2][0] = -0.008480655492356988544426874250230774675121;
                B.A[2][1] = 0.3354806554923569885444268742502307746751;
                B.A[3][0] = 2.897153057105493432130432594192938764925;
                B.A[3][1] = -6.359448489975074843148159912383825625953;
                B.A[3][2] = 4.362295432869581411017727318190886861028;
                B.A[4][0] = 5.325864828439256604428877920840511317836;
                B.A[4][1] = -11.74888356406282787774717033978577296189;
                B.A[4][2] = 7.495539342889836208304604784564358155659;
                B.A[4][3] = -0.0924950663617552492565020793320719161135;
                B.A[5][0] = 5.861455442946420028659251486982647890394;
                B.A[5][1] = -12.92096931784710929170611868178335939542;
                B.A[5][2] = 8.159367898576158643180400794539253485182;
                B.A[5][3] = -0.07158497328140099722453054252582973869127;
                B.A[5][4] = -0.02826905039406838290900305721271224146718;
                B.A[6][0] = 0.09646076681806522951816731316512876333712;
                B.A[6][1] = 0.01;
                B.A[6][2] = 0.479889650414499574775249532290596519913;
                B.A[6][3] = 1.379008574103741893192274821856872770756;
                B.A[6][4] = -3.290069515436080679901047585711363850116;
                B.A[6][5] = 2.324710524099773982415355918398765796109;

                B.b[0] = 0.09646076681806522951816731316512876333712;
                B.b[1] = 0.01;
                B.b[2] = 0.479889650414499574775249532290596519913;
                B.b[3] = 1.379008574103741893192274821856872770756;
                B.b[4] = -3.290069515436080679901047585711363850116;
                B.b[5] = 2.324710524099773982415355918398765796109;
                B.b[6] = 0.0;

                /* Table 1 and the Mathematica code to generate the coefficients in
                 * https://doi.org/10.1016/j.camwa.2011.06.002 have several errors. Table 1
                 * lists bHat coefficients but actually reports (b - bHat) with a sign error
                 * in the last coefficient. The Mathematica code uses bHat_7 = 1/40 when it
                 * should be 1/66. This is why these values do not match the paper. */
                B.d[0] = 0.09468075576583945807478876255758922856118;
                B.d[1] = 0.009183565540343253096776363936645313759814;
                B.d[2] = 0.4877705284247615707855642599631228241517;
                B.d[3] = 1.234297566930478985655109673884237654036;
                B.d[4] = -2.707712349983525454881109975059321670690;
                B.d[5] = 1.866628418170587035753719399566211498666;
                B.d[6] = 0.01515151515151515151515151515151515151515;

                B.c[0] = 0.0;
                B.c[1] = 0.161;
                B.c[2] = 0.327;
                B.c[3] = 0.9;
                B.c[4] = 0.9800255409045096857298102862870245954942;
                B.c[5] = 1.0;
                B.c[6] = 1.0;
            }
            Some(B)
        }
        ARKODE_CASH_KARP_6_4_5 => {
            /* Cash-Karp-ERK */
            let B = ARKodeButcherTable_Alloc(6, SUNTRUE)?;
            {
                let mut Bm = B.borrow_mut();
                let B = &mut *Bm;

                B.q = 5;
                B.p = 4;
                B.A[1][0] = 1.0 / 5.0;
                B.A[2][0] = 3.0 / 40.0;
                B.A[2][1] = 9.0 / 40.0;
                B.A[3][0] = 3.0 / 10.0;
                B.A[3][1] = -9.0 / 10.0;
                B.A[3][2] = 6.0 / 5.0;
                B.A[4][0] = -11.0 / 54.0;
                B.A[4][1] = 5.0 / 2.0;
                B.A[4][2] = -70.0 / 27.0;
                B.A[4][3] = 35.0 / 27.0;
                B.A[5][0] = 1631.0 / 55296.0;
                B.A[5][1] = 175.0 / 512.0;
                B.A[5][2] = 575.0 / 13824.0;
                B.A[5][3] = 44275.0 / 110592.0;
                B.A[5][4] = 253.0 / 4096.0;

                B.b[0] = 37.0 / 378.0;
                B.b[2] = 250.0 / 621.0;
                B.b[3] = 125.0 / 594.0;
                B.b[5] = 512.0 / 1771.0;

                B.d[0] = 2825.0 / 27648.0;
                B.d[2] = 18575.0 / 48384.0;
                B.d[3] = 13525.0 / 55296.0;
                B.d[4] = 277.0 / 14336.0;
                B.d[5] = 1.0 / 4.0;

                B.c[1] = 1.0 / 5.0;
                B.c[2] = 3.0 / 10.0;
                B.c[3] = 3.0 / 5.0;
                B.c[4] = 1.0;
                B.c[5] = 7.0 / 8.0;
            }
            Some(B)
        }
        ARKODE_FEHLBERG_6_4_5 => {
            /* Fehlberg-ERK */
            let B = ARKodeButcherTable_Alloc(6, SUNTRUE)?;
            {
                let mut Bm = B.borrow_mut();
                let B = &mut *Bm;

                B.q = 5;
                B.p = 4;
                B.A[1][0] = 1.0 / 4.0;
                B.A[2][0] = 3.0 / 32.0;
                B.A[2][1] = 9.0 / 32.0;
                B.A[3][0] = 1932.0 / 2197.0;
                B.A[3][1] = -7200.0 / 2197.0;
                B.A[3][2] = 7296.0 / 2197.0;
                B.A[4][0] = 439.0 / 216.0;
                B.A[4][1] = -8.0;
                B.A[4][2] = 3680.0 / 513.0;
                B.A[4][3] = -845.0 / 4104.0;
                B.A[5][0] = -8.0 / 27.0;
                B.A[5][1] = 2.0;
                B.A[5][2] = -3544.0 / 2565.0;
                B.A[5][3] = 1859.0 / 4104.0;
                B.A[5][4] = -11.0 / 40.0;

                B.b[0] = 16.0 / 135.0;
                B.b[2] = 6656.0 / 12825.0;
                B.b[3] = 28561.0 / 56430.0;
                B.b[4] = -9.0 / 50.0;
                B.b[5] = 2.0 / 55.0;

                B.d[0] = 25.0 / 216.0;
                B.d[2] = 1408.0 / 2565.0;
                B.d[3] = 2197.0 / 4104.0;
                B.d[4] = -1.0 / 5.0;

                B.c[1] = 1.0 / 4.0;
                B.c[2] = 3.0 / 8.0;
                B.c[3] = 12.0 / 13.0;
                B.c[4] = 1.0;
                B.c[5] = 1.0 / 2.0;
            }
            Some(B)
        }
        ARKODE_DORMAND_PRINCE_7_4_5 => {
            /* Dormand-Prince-ERK */
            let B = ARKodeButcherTable_Alloc(7, SUNTRUE)?;
            {
                let mut Bm = B.borrow_mut();
                let B = &mut *Bm;

                B.q = 5;
                B.p = 4;
                B.A[1][0] = 1.0 / 5.0;
                B.A[2][0] = 3.0 / 40.0;
                B.A[2][1] = 9.0 / 40.0;
                B.A[3][0] = 44.0 / 45.0;
                B.A[3][1] = -56.0 / 15.0;
                B.A[3][2] = 32.0 / 9.0;
                B.A[4][0] = 19372.0 / 6561.0;
                B.A[4][1] = -25360.0 / 2187.0;
                B.A[4][2] = 64448.0 / 6561.0;
                B.A[4][3] = -212.0 / 729.0;
                B.A[5][0] = 9017.0 / 3168.0;
                B.A[5][1] = -355.0 / 33.0;
                B.A[5][2] = 46732.0 / 5247.0;
                B.A[5][3] = 49.0 / 176.0;
                B.A[5][4] = -5103.0 / 18656.0;
                B.A[6][0] = 35.0 / 384.0;
                B.A[6][2] = 500.0 / 1113.0;
                B.A[6][3] = 125.0 / 192.0;
                B.A[6][4] = -2187.0 / 6784.0;
                B.A[6][5] = 11.0 / 84.0;

                B.b[0] = 35.0 / 384.0;
                B.b[2] = 500.0 / 1113.0;
                B.b[3] = 125.0 / 192.0;
                B.b[4] = -2187.0 / 6784.0;
                B.b[5] = 11.0 / 84.0;

                B.d[0] = 5179.0 / 57600.0;
                B.d[2] = 7571.0 / 16695.0;
                B.d[3] = 393.0 / 640.0;
                B.d[4] = -92097.0 / 339200.0;
                B.d[5] = 187.0 / 2100.0;
                B.d[6] = 1.0 / 40.0;

                B.c[1] = 1.0 / 5.0;
                B.c[2] = 3.0 / 10.0;
                B.c[3] = 4.0 / 5.0;
                B.c[4] = 8.0 / 9.0;
                B.c[5] = 1.0;
                B.c[6] = 1.0;
            }
            Some(B)
        }
        ARKODE_ARK548L2SA_ERK_8_4_5 => {
            /* ARK5(4)8L[2]SA-ERK */
            let B = ARKodeButcherTable_Alloc(8, SUNTRUE)?;
            {
                let mut Bm = B.borrow_mut();
                let B = &mut *Bm;

                B.q = 5;
                B.p = 4;
                B.A[1][0] = 41.0 / 100.0;
                B.A[2][0] = 367902744464.0 / 2072280473677.0;
                B.A[2][1] = 677623207551.0 / 8224143866563.0;
                B.A[3][0] = 1268023523408.0 / 10340822734521.0;
                B.A[3][2] = 1029933939417.0 / 13636558850479.0;
                B.A[4][0] = 14463281900351.0 / 6315353703477.0;
                B.A[4][2] = 66114435211212.0 / 5879490589093.0;
                B.A[4][3] = -54053170152839.0 / 4284798021562.0;
                B.A[5][0] = 14090043504691.0 / 34967701212078.0;
                B.A[5][2] = 15191511035443.0 / 11219624916014.0;
                B.A[5][3] = -18461159152457.0 / 12425892160975.0;
                B.A[5][4] = -281667163811.0 / 9011619295870.0;
                B.A[6][0] = 19230459214898.0 / 13134317526959.0;
                B.A[6][2] = 21275331358303.0 / 2942455364971.0;
                B.A[6][3] = -38145345988419.0 / 4862620318723.0;
                B.A[6][4] = -1.0 / 8.0;
                B.A[6][5] = -1.0 / 8.0;
                B.A[7][0] = -19977161125411.0 / 11928030595625.0;
                B.A[7][2] = -40795976796054.0 / 6384907823539.0;
                B.A[7][3] = 177454434618887.0 / 12078138498510.0;
                B.A[7][4] = 782672205425.0 / 8267701900261.0;
                B.A[7][5] = -69563011059811.0 / 9646580694205.0;
                B.A[7][6] = 7356628210526.0 / 4942186776405.0;

                B.b[0] = -872700587467.0 / 9133579230613.0;
                B.b[3] = 22348218063261.0 / 9555858737531.0;
                B.b[4] = -1143369518992.0 / 8141816002931.0;
                B.b[5] = -39379526789629.0 / 19018526304540.0;
                B.b[6] = 32727382324388.0 / 42900044865799.0;
                B.b[7] = 41.0 / 200.0;

                B.d[0] = -975461918565.0 / 9796059967033.0;
                B.d[3] = 78070527104295.0 / 32432590147079.0;
                B.d[4] = -548382580838.0 / 3424219808633.0;
                B.d[5] = -33438840321285.0 / 15594753105479.0;
                B.d[6] = 3629800801594.0 / 4656183773603.0;
                B.d[7] = 4035322873751.0 / 18575991585200.0;

                B.c[1] = 41.0 / 100.0;
                B.c[2] = 2935347310677.0 / 11292855782101.0;
                B.c[3] = 1426016391358.0 / 7196633302097.0;
                B.c[4] = 92.0 / 100.0;
                B.c[5] = 24.0 / 100.0;
                B.c[6] = 3.0 / 5.0;
                B.c[7] = 1.0;
            }
            Some(B)
        }
        ARKODE_ARK548L2SAb_ERK_8_4_5 => {
            /* ARK5(4)8L[2]SAb-ERK */
            let B = ARKodeButcherTable_Alloc(8, SUNTRUE)?;
            {
                let mut Bm = B.borrow_mut();
                let B = &mut *Bm;

                B.q = 5;
                B.p = 4;
                B.A[1][0] = 4.0 / 9.0;
                B.A[2][0] = 1.0 / 9.0;
                B.A[2][1] = 1183333538310.0 / 1827251437969.0;
                B.A[3][0] = 895379019517.0 / 9750411845327.0;
                B.A[3][1] = 477606656805.0 / 13473228687314.0;
                B.A[3][2] = -112564739183.0 / 9373365219272.0;
                B.A[4][0] = -4458043123994.0 / 13015289567637.0;
                B.A[4][1] = -2500665203865.0 / 9342069639922.0;
                B.A[4][2] = 983347055801.0 / 8893519644487.0;
                B.A[4][3] = 2185051477207.0 / 2551468980502.0;
                B.A[5][0] = -167316361917.0 / 17121522574472.0;
                B.A[5][1] = 1605541814917.0 / 7619724128744.0;
                B.A[5][2] = 991021770328.0 / 13052792161721.0;
                B.A[5][3] = 2342280609577.0 / 11279663441611.0;
                B.A[5][4] = 3012424348531.0 / 12792462456678.0;
                B.A[6][0] = 6680998715867.0 / 14310383562358.0;
                B.A[6][1] = 5029118570809.0 / 3897454228471.0;
                B.A[6][2] = 2415062538259.0 / 6382199904604.0;
                B.A[6][3] = -3924368632305.0 / 6964820224454.0;
                B.A[6][4] = -4331110370267.0 / 15021686902756.0;
                B.A[6][5] = -3944303808049.0 / 11994238218192.0;
                B.A[7][0] = 2193717860234.0 / 3570523412979.0;
                B.A[7][1] = 2193717860234.0 / 3570523412979.0;
                B.A[7][2] = 5952760925747.0 / 18750164281544.0;
                B.A[7][3] = -4412967128996.0 / 6196664114337.0;
                B.A[7][4] = 4151782504231.0 / 36106512998704.0;
                B.A[7][5] = 572599549169.0 / 6265429158920.0;
                B.A[7][6] = -457874356192.0 / 11306498036315.0;

                B.b[2] = 3517720773327.0 / 20256071687669.0;
                B.b[3] = 4569610470461.0 / 17934693873752.0;
                B.b[4] = 2819471173109.0 / 11655438449929.0;
                B.b[5] = 3296210113763.0 / 10722700128969.0;
                B.b[6] = -1142099968913.0 / 5710983926999.0;
                B.b[7] = 2.0 / 9.0;

                B.d[2] = 520639020421.0 / 8300446712847.0;
                B.d[3] = 4550235134915.0 / 17827758688493.0;
                B.d[4] = 1482366381361.0 / 6201654941325.0;
                B.d[5] = 5551607622171.0 / 13911031047899.0;
                B.d[6] = -5266607656330.0 / 36788968843917.0;
                B.d[7] = 1074053359553.0 / 5740751784926.0;

                B.c[1] = 4.0 / 9.0;
                B.c[2] = 6456083330201.0 / 8509243623797.0;
                B.c[3] = 1632083962415.0 / 14158861528103.0;
                B.c[4] = 6365430648612.0 / 17842476412687.0;
                B.c[5] = 18.0 / 25.0;
                B.c[6] = 191.0 / 200.0;
                B.c[7] = 1.0;
            }
            Some(B)
        }
        ARKODE_VERNER_8_5_6 => {
            /* Verner-6-5 */
            let B = ARKodeButcherTable_Alloc(8, SUNTRUE)?;
            {
                let mut Bm = B.borrow_mut();
                let B = &mut *Bm;

                B.q = 6;
                B.p = 5;
                B.A[1][0] = 1.0 / 6.0;
                B.A[2][0] = 4.0 / 75.0;
                B.A[2][1] = 16.0 / 75.0;
                B.A[3][0] = 5.0 / 6.0;
                B.A[3][1] = -8.0 / 3.0;
                B.A[3][2] = 5.0 / 2.0;
                B.A[4][0] = -165.0 / 64.0;
                B.A[4][1] = 55.0 / 6.0;
                B.A[4][2] = -425.0 / 64.0;
                B.A[4][3] = 85.0 / 96.0;
                B.A[5][0] = 12.0 / 5.0;
                B.A[5][1] = -8.0;
                B.A[5][2] = 4015.0 / 612.0;
                B.A[5][3] = -11.0 / 36.0;
                B.A[5][4] = 88.0 / 255.0;
                B.A[6][0] = -8263.0 / 15000.0;
                B.A[6][1] = 124.0 / 75.0;
                B.A[6][2] = -643.0 / 680.0;
                B.A[6][3] = -81.0 / 250.0;
                B.A[6][4] = 2484.0 / 10625.0;
                B.A[7][0] = 3501.0 / 1720.0;
                B.A[7][1] = -300.0 / 43.0;
                B.A[7][2] = 297275.0 / 52632.0;
                B.A[7][3] = -319.0 / 2322.0;
                B.A[7][4] = 24068.0 / 84065.0;
                B.A[7][6] = 3850.0 / 26703.0;

                B.b[0] = 3.0 / 40.0;
                B.b[2] = 875.0 / 2244.0;
                B.b[3] = 23.0 / 72.0;
                B.b[4] = 264.0 / 1955.0;
                B.b[6] = 125.0 / 11592.0;
                B.b[7] = 43.0 / 616.0;

                B.d[0] = 13.0 / 160.0;
                B.d[2] = 2375.0 / 5984.0;
                B.d[3] = 5.0 / 16.0;
                B.d[4] = 12.0 / 85.0;
                B.d[5] = 3.0 / 44.0;

                B.c[0] = 0.0;
                B.c[1] = 1.0 / 6.0;
                B.c[2] = 4.0 / 15.0;
                B.c[3] = 2.0 / 3.0;
                B.c[4] = 5.0 / 6.0;
                B.c[5] = 1.0;
                B.c[6] = 1.0 / 15.0;
                B.c[7] = 1.0;
            }
            Some(B)
        }
        ARKODE_VERNER_9_5_6 => {
            /* Verner-6-5b (DOI: 10.1007/s11075-009-9290-3) */
            let B = ARKodeButcherTable_Alloc(9, SUNTRUE)?;
            {
                let mut Bm = B.borrow_mut();
                let B = &mut *Bm;

                B.q = 6;
                B.p = 5;

                B.A[1][0] = 0.06;
                B.A[2][0] = 0.01923996296296296296296296296296296296296;
                B.A[2][1] = 0.07669337037037037037037037037037037037037;
                B.A[3][0] = 0.035975;
                B.A[3][2] = 0.107925;
                B.A[4][0] = 1.318683415233148260919747276431735612861;
                B.A[4][2] = -5.042058063628562225427761634715637693344;
                B.A[4][3] = 4.220674648395413964508014358283902080483;
                B.A[5][0] = -41.87259166432751461803757780644346812905;
                B.A[5][2] = 159.4325621631374917700365669070346830453;
                B.A[5][3] = -122.1192135650100309202516203389242140663;
                B.A[5][4] = 5.531743066200053768252631238332999150076;
                B.A[6][0] = -54.43015693531650433250642051294142461271;
                B.A[6][2] = 207.0672513650184644273657173866509835987;
                B.A[6][3] = -158.6108137845899991828742424365058599469;
                B.A[6][4] = 6.991816585950242321992597280791793907096;
                B.A[6][5] = -0.01859723106220323397765171799549294623692;
                B.A[7][0] = -54.66374178728197680241215648050386959351;
                B.A[7][2] = 207.9528062553893734515824816699834244238;
                B.A[7][3] = -159.2889574744995071508959805871426654216;
                B.A[7][4] = 7.018743740796944434698170760964252490817;
                B.A[7][5] = -0.01833878590504572306472782005141738268361;
                B.A[7][6] = -0.000511948499788209907787543249724516839584;
                B.A[8][0] = 0.0343895786835703600927882012472832238652;
                B.A[8][3] = 0.2582624555633503404659558098586120858767;
                B.A[8][4] = 0.4209371189673537150642551514069801967032;
                B.A[8][5] = 4.405396469669310170148836816197095664891;
                B.A[8][6] = -176.4831190242986576151740942499002125029;
                B.A[8][7] = 172.3641334014150730294022582711902413315;

                B.b[0] = 0.0343895786835703600927882012472832238652;
                B.b[3] = 0.2582624555633503404659558098586120858767;
                B.b[4] = 0.4209371189673537150642551514069801967032;
                B.b[5] = 4.405396469669310170148836816197095664891;
                B.b[6] = -176.4831190242986576151740942499002125029;
                B.b[7] = 172.3641334014150730294022582711902413315;

                B.d[0] = 0.04909967648382489730906854927971225836479;
                B.d[3] = 0.2251112229516524153401395320539875329485;
                B.d[4] = 0.4694682253029562039431948525047387412553;
                B.d[5] = 0.8065792249988867707634161808995217981443;
                B.d[7] = -0.6071194891777959797672951465256217122488;
                B.d[8] = 0.05686113944047569241147603178766138153594;

                B.c[1] = 0.06;
                B.c[2] = 0.09593333333333333333333333333333333333333;
                B.c[3] = 0.1439;
                B.c[4] = 0.4973;
                B.c[5] = 0.9725;
                B.c[6] = 0.9995;
                B.c[7] = 1.0;
                B.c[8] = 1.0;
            }
            Some(B)
        }
        ARKODE_VERNER_10_6_7 => {
            /* Verner-7-6 (DOI: 10.1007/s11075-009-9290-3) */
            let B = ARKodeButcherTable_Alloc(10, SUNTRUE)?;
            {
                let mut Bm = B.borrow_mut();
                let B = &mut *Bm;

                B.q = 7;
                B.p = 6;

                B.A[1][0] = 0.005;
                B.A[2][0] = -1.076790123456790123456790123456790123457;
                B.A[2][1] = 1.185679012345679012345679012345679012346;
                B.A[3][0] = 0.04083333333333333333333333333333333333333;
                B.A[3][2] = 0.1225;
                B.A[4][0] = 0.6389139236255726780508121615993336109954;
                B.A[4][2] = -2.455672638223656809662640566430653894211;
                B.A[4][3] = 2.272258714598084131611828404831320283215;
                B.A[5][0] = -2.661577375018757131119259297861818119279;
                B.A[5][2] = 10.80451388645613769565396655365532838482;
                B.A[5][3] = -8.353914657396199411968048547819291691541;
                B.A[5][4] = 0.8204875949566569791420417341743839209619;
                B.A[6][0] = 6.067741434696770992718360183877276714679;
                B.A[6][2] = -24.71127363591108579734203485290746001803;
                B.A[6][3] = 20.42751793078889394045773111748346612697;
                B.A[6][4] = -1.906157978816647150624096784352757010879;
                B.A[6][5] = 1.006172249242068014790040335899474187268;
                B.A[7][0] = 12.05467007625320299509109452892778311648;
                B.A[7][2] = -49.75478495046898932807257615331444758322;
                B.A[7][3] = 41.14288863860467663259698416710157354209;
                B.A[7][4] = -4.461760149974004185641911603484815375051;
                B.A[7][5] = 2.042334822239174959821717077708608543738;
                B.A[7][6] = -0.09834843665406107379530801693870224403537;
                B.A[8][0] = 10.13814652288180787641845141981689030769;
                B.A[8][2] = -42.64113603171750214622846006736635730625;
                B.A[8][3] = 35.76384003992257007135021178023160054034;
                B.A[8][4] = -4.34802284039290765334037029690824594371;
                B.A[8][5] = 2.009862268377035895441943593011827554771;
                B.A[8][6] = 0.348749046033827240595382285305314587914;
                B.A[8][7] = -0.2714390051048312842371587140910297407572;
                B.A[9][0] = -45.03007203429867712435322405073769635151;
                B.A[9][2] = 187.3272437654588840752418206154201997384;
                B.A[9][3] = -154.0288236935018690596728621034510402582;
                B.A[9][4] = 18.56465306347536233859492332958439136765;
                B.A[9][5] = -7.141809679295078854925420496823551192821;
                B.A[9][6] = 1.308808578161378625114762706007696696508;

                B.b[0] = 0.04715561848627222170431765108838175679569;
                B.b[3] = 0.2575056429843415189596436101037687580986;
                B.b[4] = 0.2621665397741262047713863095764527711129;
                B.b[5] = 0.1521609265673855740323133199165117535523;
                B.b[6] = 0.4939969170032484246907175893227876844296;
                B.b[7] = -0.2943031171403250441557244744092703429139;
                B.b[8] = 0.08131747232495109999734599440136761892478;

                B.d[0] = 0.04460860660634117628731817597479197781432;
                B.d[3] = 0.2671640378571372680509102260943837899738;
                B.d[4] = 0.2201018300177293019979715776650753096323;
                B.d[5] = 0.2188431703143156830983120833512893824578;
                B.d[6] = 0.2289871705411202883378173889763552365362;
                B.d[9] = 0.02029518466335628222767054793810430358554;

                B.c[1] = 0.005;
                B.c[2] = 0.1088888888888888888888888888888888888889;
                B.c[3] = 0.1633333333333333333333333333333333333333;
                B.c[4] = 0.4555;
                B.c[5] = 0.6095094489978381317087004421486024949638;
                B.c[6] = 0.884;
                B.c[7] = 0.925;
                B.c[8] = 1.0;
                B.c[9] = 1.0;
            }
            Some(B)
        }
        ARKODE_VERNER_13_7_8 => {
            /* Verner-8-7 (DOI: 10.1007/s11075-009-9290-3) */
            let B = ARKodeButcherTable_Alloc(13, SUNTRUE)?;
            {
                let mut Bm = B.borrow_mut();
                let B = &mut *Bm;

                B.q = 8;
                B.p = 7;

                B.A[1][0] = 0.05;
                B.A[2][0] = -0.0069931640625;
                B.A[2][1] = 0.1135556640625;
                B.A[3][0] = 0.0399609375;
                B.A[3][2] = 0.1198828125;
                B.A[4][0] = 0.3613975628004575124052940721184028345129;
                B.A[4][2] = -1.341524066700492771819987788202715834917;
                B.A[4][3] = 1.370126503900035259414693716084313000404;
                B.A[5][0] = 0.0490472027972027972027972027972027972028;
                B.A[5][3] = 0.2350972042214404739862988335493427143122;
                B.A[5][4] = 0.180855592981356728810903963653454488485;
                B.A[6][0] = 0.06169289044289044289044289044289044289044;
                B.A[6][3] = 0.1123656831464027662262557035130015442303;
                B.A[6][4] = -0.03885046071451366767049048108111244567456;
                B.A[6][5] = 0.01979188712522045855379188712522045855379;
                B.A[7][0] = -1.767630240222326875735597119572145586714;
                B.A[7][3] = -62.5;
                B.A[7][4] = -6.061889377376669100821361459659331999758;
                B.A[7][5] = 5.650823198222763138561298030600840174201;
                B.A[7][6] = 65.62169641937623283799566054863063741227;
                B.A[8][0] = -1.180945066554970799825116282628297957882;
                B.A[8][3] = -41.50473441114320841606641502701994225874;
                B.A[8][4] = -4.434438319103725011225169229846100211776;
                B.A[8][5] = 4.260408188586133024812193710744693240761;
                B.A[8][6] = 43.75364022446171584987676829438379303004;
                B.A[8][7] = 0.00787142548991231068744647504422630755086;
                B.A[9][0] = -1.281405999441488405459510291182054246266;
                B.A[9][3] = -45.04713996013986630220754257136007322267;
                B.A[9][4] = -4.731362069449576477311464265491282810943;
                B.A[9][5] = 4.514967016593807841185851584597240996214;
                B.A[9][6] = 47.44909557172985134869022392235929015114;
                B.A[9][7] = 0.01059228297111661135687393955516542875228;
                B.A[9][8] = -0.005746842263844616254432318478286296232021;
                B.A[10][0] = -1.724470134262485191756709817484481861731;
                B.A[10][3] = -60.92349008483054016518434619253765246063;
                B.A[10][4] = -5.95151837622239245520283276706185486829;
                B.A[10][5] = 5.556523730698456235979791650843592496839;
                B.A[10][6] = 63.98301198033305336837536378635995939281;
                B.A[10][7] = 0.01464202825041496159275921391759452676003;
                B.A[10][8] = 0.06460408772358203603621865144977650714892;
                B.A[10][9] = -0.07930323169008878984024452548693373291447;
                B.A[11][0] = -3.301622667747079016353994789790983625569;
                B.A[11][3] = -118.011272359752508566692330395789886851;
                B.A[11][4] = -10.14142238845611248642783916034510897595;
                B.A[11][5] = 9.139311332232057923544012273556827000619;
                B.A[11][6] = 123.3759428284042683684847180986501894364;
                B.A[11][7] = 4.623244378874580474839807625067630924792;
                B.A[11][8] = -3.383277738068201923652550971536811240814;
                B.A[11][9] = 4.527592100324618189451265339351129035325;
                B.A[11][10] = -5.828495485811622963193088019162985703755;
                B.A[12][0] = -3.039515033766309030040102851821200251056;
                B.A[12][3] = -109.2608680894176254686444192322164623352;
                B.A[12][4] = -9.290642497400293449717665542656897549158;
                B.A[12][5] = 8.430504981764911142134299253836167803454;
                B.A[12][6] = 114.2010010378331313557424041095523427476;
                B.A[12][7] = -0.9637271342145479358162375658987901652762;
                B.A[12][8] = -5.034884088802189791198680336183332323118;
                B.A[12][9] = 5.958130824002923177540402165388172072794;

                B.b[0] = 0.04427989419007951074716746668098518862111;
                B.b[5] = 0.3541049391724448744815552028733568354121;
                B.b[6] = 0.2479692154956437828667629415370663023884;
                B.b[7] = -15.69420203883808405099207034271191213468;
                B.b[8] = 25.08406496555856261343930031237186278518;
                B.b[9] = -31.73836778626027646833156112007297739997;
                B.b[10] = 22.93828327398878395231483560344797018313;
                B.b[11] = -0.2361324633071542145259900641263517600737;

                B.d[0] = 0.04431261522908979212486436510209029764893;
                B.d[5] = 0.3546095642343226447863179350895055038855;
                B.d[6] = 0.2478480431366653069619986721504458660017;
                B.d[7] = 4.448134732475784492725128317159648871312;
                B.d[8] = 19.84688636611873369930932399297687935291;
                B.d[9] = -23.58162337746561841969517960870394965085;
                B.d[12] = -0.360167943728977516212453673774620240911;

                B.c[1] = 0.05;
                B.c[2] = 0.1065625;
                B.c[3] = 0.15984375;
                B.c[4] = 0.39;
                B.c[5] = 0.465;
                B.c[6] = 0.155;
                B.c[7] = 0.943;
                B.c[8] = 0.901802041735856958259707940678372149956;
                B.c[9] = 0.909;
                B.c[10] = 0.94;
                B.c[11] = 1.0;
                B.c[12] = 1.0;
            }
            Some(B)
        }
        ARKODE_VERNER_16_8_9 => {
            /* Verner-9-8 (DOI: 10.1007/s11075-009-9290-3) */
            let B = ARKodeButcherTable_Alloc(16, SUNTRUE)?;
            {
                let mut Bm = B.borrow_mut();
                let B = &mut *Bm;

                B.q = 9;
                B.p = 8;

                B.A[1][0] = 0.3462e-1;
                B.A[2][0] = -0.389335438857287327017042687229284478532e-1;
                B.A[2][1] = 0.1359578945245091786499878854939346230295;
                B.A[3][0] = 0.3638413148954266723060635628912731569111e-1;
                B.A[3][2] = 0.1091523944686280016918190688673819470733;
                B.A[4][0] = 2.025763914393969636805657604282571047511;
                B.A[4][2] = -7.638023836496292020387602153091964592952;
                B.A[4][3] = 6.173259922102322383581944548809393545442;
                B.A[5][0] = 0.5112275589406060872792270881648288397197e-1;
                B.A[5][3] = 0.1770823794555021537929910813839068684087;
                B.A[5][4] = 0.80277624092225014536138698108025283759e-3;
                B.A[6][0] = 0.1316006357975216279279871693164256985334;
                B.A[6][3] = -0.2957276252669636417685183174672273730699;
                B.A[6][4] = 0.878137803564295237421124704053886667082e-1;
                B.A[6][5] = 0.6213052975225274774321435005639430026100;
                B.A[7][0] = 0.7166666666666666666666666666666666666667e-1;
                B.A[7][5] = 0.3305533578915319409260346730051472207728;
                B.A[7][6] = 0.2427799754418013924072986603281861125606;
                B.A[8][0] = 0.7180664062500000000000000000000000000000e-1;
                B.A[8][5] = 0.3294380283228177160744825466257672816401;
                B.A[8][6] = 0.1165190029271822839255174533742327183599;
                B.A[8][7] = -0.3401367187500000000000000000000000000000e-1;
                B.A[9][0] = 0.4836757646340646986611287718844085773549e-1;
                B.A[9][5] = 0.3928989925676163974333190042057047002852e-1;
                B.A[9][6] = 0.1054740945890344608263649267140088017604;
                B.A[9][7] = -0.2143865284648312665982642293830533996214e-1;
                B.A[9][8] = -0.1041229174627194437759832813847147895623;
                B.A[10][0] = -0.2664561487201478635337289243849737340534e-1;
                B.A[10][5] = 0.3333333333333333333333333333333333333333e-1;
                B.A[10][6] = -0.1631072244872467239162704487554706387141;
                B.A[10][7] = 0.3396081684127761199487954930015522928244e-1;
                B.A[10][8] = 0.1572319413814626097110769806810024118077;
                B.A[10][9] = 0.2152267478031879552303534778794770376960;
                B.A[11][0] = 0.3689009248708622334786359863227633989718e-1;
                B.A[11][5] = -0.1465181576725542928653609891758501156785;
                B.A[11][6] = 0.2242577768172024345345469822625833796001;
                B.A[11][7] = 0.2294405717066072637090897902753790803034e-1;
                B.A[11][8] = -0.35850052905728761357394424889330334334e-2;
                B.A[11][9] = 0.8669223316444385506869203619044453906053e-1;
                B.A[11][10] = 0.4383840651968337846196219974168630120572;
                B.A[12][0] = -0.4866012215113340846662212357570395295088;
                B.A[12][5] = -6.304602650282852990657772792012007122988;
                B.A[12][6] = -0.281245618289472564778284183790118418111;
                B.A[12][7] = -2.679019236219849057687906597489223155566;
                B.A[12][8] = 0.518815663924157511565311164615012522024;
                B.A[12][9] = 1.365353187603341710683633635235238678626;
                B.A[12][10] = 5.885091088503946585721274891680604830712;
                B.A[12][11] = 2.802808786272062889819965117517532194812;
                B.A[13][0] = 0.4185367457753471441471025246471931649633;
                B.A[13][5] = 6.724547581906459363100870806514855026676;
                B.A[13][6] = -0.425444280164611790606983409697113064616;
                B.A[13][7] = 3.343279153001265577811816947557982637749;
                B.A[13][8] = 0.617081663117537759528421117507709784737;
                B.A[13][9] = -0.929966123939932833937749523988800852013;
                B.A[13][10] = -6.099948804751010722472962837945508844846;
                B.A[13][11] = -3.002206187889399044804158084895173690015;
                B.A[13][12] = 0.2553202529443445472336424602988558373637;
                B.A[14][0] = -0.779374086122884664644623040843840506343;
                B.A[14][5] = -13.93734253810777678786523664804936051203;
                B.A[14][6] = 1.252048853379357320949735183924200895136;
                B.A[14][7] = -14.69150040801686878191527989293072091588;
                B.A[14][8] = -0.494705058533141685655191992136962873577;
                B.A[14][9] = 2.242974909146236657906984549543692874755;
                B.A[14][10] = 13.36789380382864375813864978592679139881;
                B.A[14][11] = 14.39665048665068644512236935340272139005;
                B.A[14][12] = -0.7975813331776800379127866056663258667437;
                B.A[14][13] = 0.4409353709534277758753793068298041158235;
                B.A[15][0] = 2.058051337466886442151242368989994043993;
                B.A[15][5] = 22.35793772796803295519317565842520212899;
                B.A[15][6] = 0.90949810997556332745009198137971890783;
                B.A[15][7] = 35.89110098240264104710550686568482456493;
                B.A[15][8] = -3.442515027624453437985000403608480262211;
                B.A[15][9] = -4.865481358036368826566013387928704014496;
                B.A[15][10] = -18.90980381354342625688427480879773032857;
                B.A[15][11] = -34.26354448030451782929251177395134170515;
                B.A[15][12] = 1.264756521695642578827783499806516664686;

                B.b[0] = 0.1461197685842315252051541915018784713459e-1;
                B.b[7] = -0.3915211862331339089410228267288242030810;
                B.b[8] = 0.2310932500289506415909675644868993669908;
                B.b[9] = 0.1274766769992852382560589467488989175618;
                B.b[10] = 0.2246434176204157731566981937082069688984;
                B.b[11] = 0.5684352689748512932705226972873692126743;
                B.b[12] = 0.5825871557215827200814768021863420902155e-1;
                B.b[13] = 0.1364317403482215641609022744494239843327;
                B.b[14] = 0.3057013983082797397721005067920369646664e-1;

                B.d[0] = 0.1996996514886773085518508418098868756464e-1;
                B.d[7] = 2.191499304949330054530747099310837524864;
                B.d[8] = 0.8857071848208438030833722031786358862953e-1;
                B.d[9] = 0.1140560234865965622484956605091432032674;
                B.d[10] = 0.2533163805345107065564577734569651977347;
                B.d[11] = -2.056564386240941011158999594595981300493;
                B.d[12] = 0.3408096799013119935160094894224543812830;
                B.d[15] = 0.4834231373823958314376726739772871714902e-1;

                B.c[1] = 0.3462e-1;
                B.c[2] = 0.9702435063878044594828361677100617517633e-1;
                B.c[3] = 0.1455365259581706689224254251565092627645;
                B.c[4] = 0.561;
                B.c[5] = 0.2290079115904850126662751771814700052182;
                B.c[6] = 0.5449920884095149873337248228185299947818;
                B.c[7] = 0.645;
                B.c[8] = 0.48375;
                B.c[9] = 0.6757e-1;
                B.c[10] = 0.25;
                B.c[11] = 0.6590650618730998549405331618649220295334;
                B.c[12] = 0.8206;
                B.c[13] = 0.9012;
                B.c[14] = 1.0;
                B.c[15] = 1.0;
            }
            Some(B)
        }
        ARKODE_FEHLBERG_13_7_8 => {
            /* Fehlberg-8-7 */
            let B = ARKodeButcherTable_Alloc(13, SUNTRUE)?;
            {
                let mut Bm = B.borrow_mut();
                let B = &mut *Bm;

                B.q = 8;
                B.p = 7;
                B.A[1][0] = 2.0 / 27.0;
                B.A[2][0] = 1.0 / 36.0;
                B.A[2][1] = 1.0 / 12.0;
                B.A[3][0] = 1.0 / 24.0;
                B.A[3][2] = 1.0 / 8.0;
                B.A[4][0] = 5.0 / 12.0;
                B.A[4][2] = -25.0 / 16.0;
                B.A[4][3] = 25.0 / 16.0;
                B.A[5][0] = 1.0 / 20.0;
                B.A[5][3] = 1.0 / 4.0;
                B.A[5][4] = 1.0 / 5.0;
                B.A[6][0] = -25.0 / 108.0;
                B.A[6][3] = 125.0 / 108.0;
                B.A[6][4] = -65.0 / 27.0;
                B.A[6][5] = 125.0 / 54.0;
                B.A[7][0] = 31.0 / 300.0;
                B.A[7][4] = 61.0 / 225.0;
                B.A[7][5] = -2.0 / 9.0;
                B.A[7][6] = 13.0 / 900.0;
                B.A[8][0] = 2.0;
                B.A[8][3] = -53.0 / 6.0;
                B.A[8][4] = 704.0 / 45.0;
                B.A[8][5] = -107.0 / 9.0;
                B.A[8][6] = 67.0 / 90.0;
                B.A[8][7] = 3.0;
                B.A[9][0] = -91.0 / 108.0;
                B.A[9][3] = 23.0 / 108.0;
                B.A[9][4] = -976.0 / 135.0;
                B.A[9][5] = 311.0 / 54.0;
                B.A[9][6] = -19.0 / 60.0;
                B.A[9][7] = 17.0 / 6.0;
                B.A[9][8] = -1.0 / 12.0;
                B.A[10][0] = 2383.0 / 4100.0;
                B.A[10][3] = -341.0 / 164.0;
                B.A[10][4] = 4496.0 / 1025.0;
                B.A[10][5] = -301.0 / 82.0;
                B.A[10][6] = 2133.0 / 4100.0;
                B.A[10][7] = 45.0 / 82.0;
                B.A[10][8] = 45.0 / 164.0;
                B.A[10][9] = 18.0 / 41.0;
                B.A[11][0] = 3.0 / 205.0;
                B.A[11][5] = -6.0 / 41.0;
                B.A[11][6] = -3.0 / 205.0;
                B.A[11][7] = -3.0 / 41.0;
                B.A[11][8] = 3.0 / 41.0;
                B.A[11][9] = 6.0 / 41.0;
                B.A[12][0] = -1777.0 / 4100.0;
                B.A[12][3] = -341.0 / 164.0;
                B.A[12][4] = 4496.0 / 1025.0;
                B.A[12][5] = -289.0 / 82.0;
                B.A[12][6] = 2193.0 / 4100.0;
                B.A[12][7] = 51.0 / 82.0;
                B.A[12][8] = 33.0 / 164.0;
                B.A[12][9] = 12.0 / 41.0;
                B.A[12][11] = 1.0;

                B.b[5] = 34.0 / 105.0;
                B.b[6] = 9.0 / 35.0;
                B.b[7] = 9.0 / 35.0;
                B.b[8] = 9.0 / 280.0;
                B.b[9] = 9.0 / 280.0;
                B.b[11] = 41.0 / 840.0;
                B.b[12] = 41.0 / 840.0;

                B.d[0] = 41.0 / 840.0;
                B.d[5] = 34.0 / 105.0;
                B.d[6] = 9.0 / 35.0;
                B.d[7] = 9.0 / 35.0;
                B.d[8] = 9.0 / 280.0;
                B.d[9] = 9.0 / 280.0;
                B.d[10] = 41.0 / 840.0;

                B.c[1] = 2.0 / 27.0;
                B.c[2] = 1.0 / 9.0;
                B.c[3] = 1.0 / 6.0;
                B.c[4] = 5.0 / 12.0;
                B.c[5] = 1.0 / 2.0;
                B.c[6] = 5.0 / 6.0;
                B.c[7] = 1.0 / 6.0;
                B.c[8] = 2.0 / 3.0;
                B.c[9] = 1.0 / 3.0;
                B.c[10] = 1.0;
                B.c[12] = 1.0;
            }
            Some(B)
        }
        ARKODE_KNOTH_WOLKE_3_3 => {
            /* Knoth-Wolke-ERK */
            let B = ARKodeButcherTable_Alloc(3, SUNFALSE)?;
            {
                let mut Bm = B.borrow_mut();
                let B = &mut *Bm;

                B.q = 3;
                B.p = 0;
                B.A[1][0] = 1.0 / 3.0;
                B.A[2][0] = -3.0 / 16.0;
                B.A[2][1] = 15.0 / 16.0;

                B.b[0] = 1.0 / 6.0;
                B.b[1] = 3.0 / 10.0;
                B.b[2] = 8.0 / 15.0;

                B.d = Vec::new();

                B.c[1] = 1.0 / 3.0;
                B.c[2] = 3.0 / 4.0;
            }
            Some(B)
        }

        _ => {
            arkProcessError(
                None,
                ARK_ILL_INPUT,
                line!() as i32,
                "ARKodeButcherTable_LoadERK",
                file!(),
                "Unknown Butcher table",
            );
            None
        }
    }
}

/*---------------------------------------------------------------
  Returns Butcher table structure for pre-set Runge Kutta methods.

  Input:  method -- string key for the desired method
  ---------------------------------------------------------------*/
pub fn ARKodeButcherTable_LoadERKByName(emethod: &str) -> Option<ARKodeButcherTable> {
    ARKodeButcherTable_LoadERK(arkButcherTableERKNameToID(emethod))
}

/*---------------------------------------------------------------
  Returns the string name for a pre-set Runge Kutta method by its ID.

  Input:  emethod -- integer key for the desired method
  ---------------------------------------------------------------*/
pub fn ARKodeButcherTable_ERKIDToName(emethod: ARKODE_ERKTableID) -> Option<&'static str> {
    /* Use X-macro to test each method name */
    match emethod {
        ARKODE_ERK_NONE => Some("ARKODE_ERK_NONE"),
        ARKODE_FORWARD_EULER_1_1 => Some("ARKODE_FORWARD_EULER_1_1"),
        ARKODE_RALSTON_3_1_2 => Some("ARKODE_RALSTON_3_1_2"),
        ARKODE_HEUN_EULER_2_1_2 => Some("ARKODE_HEUN_EULER_2_1_2"),
        ARKODE_RALSTON_EULER_2_1_2 => Some("ARKODE_RALSTON_EULER_2_1_2"),
        ARKODE_EXPLICIT_MIDPOINT_EULER_2_1_2 => Some("ARKODE_EXPLICIT_MIDPOINT_EULER_2_1_2"),
        ARKODE_ARK2_ERK_3_1_2 => Some("ARKODE_ARK2_ERK_3_1_2"),
        ARKODE_BOGACKI_SHAMPINE_4_2_3 => Some("ARKODE_BOGACKI_SHAMPINE_4_2_3"),
        ARKODE_SSP_ERK_3_1_2 => Some("ARKODE_SSP_ERK_3_1_2"),
        ARKODE_SSP_ERK_4_1_2 => Some("ARKODE_SSP_ERK_4_1_2"),
        ARKODE_ASCHER_ERK_3_1_2 => Some("ARKODE_ASCHER_ERK_3_1_2"),
        ARKODE_SSP_ERK_4_2_3 => Some("ARKODE_SSP_ERK_4_2_3"),
        ARKODE_SSP_ERK_10_3_4 => Some("ARKODE_SSP_ERK_10_3_4"),
        ARKODE_ARK324L2SA_ERK_4_2_3 => Some("ARKODE_ARK324L2SA_ERK_4_2_3"),
        ARKODE_SHU_OSHER_3_2_3 => Some("ARKODE_SHU_OSHER_3_2_3"),
        ARKODE_SSP_LSPUM_ERK_3_1_2 => Some("ARKODE_SSP_LSPUM_ERK_3_1_2"),
        ARKODE_SOFRONIOU_SPALETTA_5_3_4 => Some("ARKODE_SOFRONIOU_SPALETTA_5_3_4"),
        ARKODE_ZONNEVELD_5_3_4 => Some("ARKODE_ZONNEVELD_5_3_4"),
        ARKODE_ARK436L2SA_ERK_6_3_4 => Some("ARKODE_ARK436L2SA_ERK_6_3_4"),
        ARKODE_ARK437L2SA_ERK_7_3_4 => Some("ARKODE_ARK437L2SA_ERK_7_3_4"),
        ARKODE_SAYFY_ABURUB_6_3_4 => Some("ARKODE_SAYFY_ABURUB_6_3_4"),
        ARKODE_TSITOURAS_7_4_5 => Some("ARKODE_TSITOURAS_7_4_5"),
        ARKODE_CASH_KARP_6_4_5 => Some("ARKODE_CASH_KARP_6_4_5"),
        ARKODE_FEHLBERG_6_4_5 => Some("ARKODE_FEHLBERG_6_4_5"),
        ARKODE_DORMAND_PRINCE_7_4_5 => Some("ARKODE_DORMAND_PRINCE_7_4_5"),
        ARKODE_ARK548L2SA_ERK_8_4_5 => Some("ARKODE_ARK548L2SA_ERK_8_4_5"),
        ARKODE_ARK548L2SAb_ERK_8_4_5 => Some("ARKODE_ARK548L2SAb_ERK_8_4_5"),
        ARKODE_VERNER_8_5_6 => Some("ARKODE_VERNER_8_5_6"),
        ARKODE_VERNER_9_5_6 => Some("ARKODE_VERNER_9_5_6"),
        ARKODE_VERNER_10_6_7 => Some("ARKODE_VERNER_10_6_7"),
        ARKODE_VERNER_13_7_8 => Some("ARKODE_VERNER_13_7_8"),
        ARKODE_VERNER_16_8_9 => Some("ARKODE_VERNER_16_8_9"),
        ARKODE_FEHLBERG_13_7_8 => Some("ARKODE_FEHLBERG_13_7_8"),
        ARKODE_KNOTH_WOLKE_3_3 => Some("ARKODE_KNOTH_WOLKE_3_3"),

        _ => {
            arkProcessError(
                None,
                ARK_ILL_INPUT,
                line!() as i32,
                "ARKodeButcherTable_ERKIDToName",
                file!(),
                "Unknown Butcher table",
            );
            None
        }
    }
}

/*---------------------------------------------------------------
  Returns Butcher table ID for pre-set Runge Kutta methods.

  Input:  method -- string key for the desired method
  ---------------------------------------------------------------*/
pub fn arkButcherTableERKNameToID(emethod: &str) -> ARKODE_ERKTableID {
    /* Use X-macro to test each method name */
    if "ARKODE_ERK_NONE" == emethod {
        return ARKODE_ERK_NONE;
    }
    if "ARKODE_FORWARD_EULER_1_1" == emethod {
        return ARKODE_FORWARD_EULER_1_1;
    }
    if "ARKODE_RALSTON_3_1_2" == emethod {
        return ARKODE_RALSTON_3_1_2;
    }
    if "ARKODE_HEUN_EULER_2_1_2" == emethod {
        return ARKODE_HEUN_EULER_2_1_2;
    }
    if "ARKODE_RALSTON_EULER_2_1_2" == emethod {
        return ARKODE_RALSTON_EULER_2_1_2;
    }
    if "ARKODE_EXPLICIT_MIDPOINT_EULER_2_1_2" == emethod {
        return ARKODE_EXPLICIT_MIDPOINT_EULER_2_1_2;
    }
    if "ARKODE_ARK2_ERK_3_1_2" == emethod {
        return ARKODE_ARK2_ERK_3_1_2;
    }
    if "ARKODE_BOGACKI_SHAMPINE_4_2_3" == emethod {
        return ARKODE_BOGACKI_SHAMPINE_4_2_3;
    }
    if "ARKODE_SSP_ERK_3_1_2" == emethod {
        return ARKODE_SSP_ERK_3_1_2;
    }
    if "ARKODE_SSP_ERK_4_1_2" == emethod {
        return ARKODE_SSP_ERK_4_1_2;
    }
    if "ARKODE_ASCHER_ERK_3_1_2" == emethod {
        return ARKODE_ASCHER_ERK_3_1_2;
    }
    if "ARKODE_SSP_ERK_4_2_3" == emethod {
        return ARKODE_SSP_ERK_4_2_3;
    }
    if "ARKODE_SSP_ERK_10_3_4" == emethod {
        return ARKODE_SSP_ERK_10_3_4;
    }
    if "ARKODE_ARK324L2SA_ERK_4_2_3" == emethod {
        return ARKODE_ARK324L2SA_ERK_4_2_3;
    }
    if "ARKODE_SHU_OSHER_3_2_3" == emethod {
        return ARKODE_SHU_OSHER_3_2_3;
    }
    if "ARKODE_SSP_LSPUM_ERK_3_1_2" == emethod {
        return ARKODE_SSP_LSPUM_ERK_3_1_2;
    }
    if "ARKODE_SOFRONIOU_SPALETTA_5_3_4" == emethod {
        return ARKODE_SOFRONIOU_SPALETTA_5_3_4;
    }
    if "ARKODE_ZONNEVELD_5_3_4" == emethod {
        return ARKODE_ZONNEVELD_5_3_4;
    }
    if "ARKODE_ARK436L2SA_ERK_6_3_4" == emethod {
        return ARKODE_ARK436L2SA_ERK_6_3_4;
    }
    if "ARKODE_ARK437L2SA_ERK_7_3_4" == emethod {
        return ARKODE_ARK437L2SA_ERK_7_3_4;
    }
    if "ARKODE_SAYFY_ABURUB_6_3_4" == emethod {
        return ARKODE_SAYFY_ABURUB_6_3_4;
    }
    if "ARKODE_TSITOURAS_7_4_5" == emethod {
        return ARKODE_TSITOURAS_7_4_5;
    }
    if "ARKODE_CASH_KARP_6_4_5" == emethod {
        return ARKODE_CASH_KARP_6_4_5;
    }
    if "ARKODE_FEHLBERG_6_4_5" == emethod {
        return ARKODE_FEHLBERG_6_4_5;
    }
    if "ARKODE_DORMAND_PRINCE_7_4_5" == emethod {
        return ARKODE_DORMAND_PRINCE_7_4_5;
    }
    if "ARKODE_ARK548L2SA_ERK_8_4_5" == emethod {
        return ARKODE_ARK548L2SA_ERK_8_4_5;
    }
    if "ARKODE_ARK548L2SAb_ERK_8_4_5" == emethod {
        return ARKODE_ARK548L2SAb_ERK_8_4_5;
    }
    if "ARKODE_VERNER_8_5_6" == emethod {
        return ARKODE_VERNER_8_5_6;
    }
    if "ARKODE_VERNER_9_5_6" == emethod {
        return ARKODE_VERNER_9_5_6;
    }
    if "ARKODE_VERNER_10_6_7" == emethod {
        return ARKODE_VERNER_10_6_7;
    }
    if "ARKODE_VERNER_13_7_8" == emethod {
        return ARKODE_VERNER_13_7_8;
    }
    if "ARKODE_VERNER_16_8_9" == emethod {
        return ARKODE_VERNER_16_8_9;
    }
    if "ARKODE_FEHLBERG_13_7_8" == emethod {
        return ARKODE_FEHLBERG_13_7_8;
    }
    if "ARKODE_KNOTH_WOLKE_3_3" == emethod {
        return ARKODE_KNOTH_WOLKE_3_3;
    }

    arkProcessError(
        None,
        ARK_ILL_INPUT,
        line!() as i32,
        "arkButcherTableERKNameToID",
        file!(),
        "Unknown Butcher table",
    );

    ARKODE_ERK_NONE
}

/*---------------------------------------------------------------
  EOF
  ---------------------------------------------------------------*/
