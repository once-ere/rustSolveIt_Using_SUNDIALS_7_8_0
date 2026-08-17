//! Port of `src/arkode/arkode_splittingstep_coefficients.c` (+ the
//! `SplittingStepCoefficientsMem` struct and `ARKODE_SplittingCoefficientsID`
//! enum of `include/arkode/arkode_splittingstep.h`, and the X-macro bodies of
//! `src/arkode/arkode_splittingstep_coefficients.def`, which fold in here
//! exactly as the workspace rule prescribes).
//!
//! Handle model (ARCHITECTURE.md): `SplittingStepCoefficients =
//! Rc<RefCell<SplittingStepCoefficientsMem>>` — the same shape the contract
//! fixes for `ARKodeButcherTable`. Cloning the `Rc` is C's pointer copy;
//! `SplittingStepCoefficients_Destroy` sets the caller's slot to `None` and
//! drops one reference (storage is released once the last clone goes away,
//! where C `free`s immediately and leaves every other copy dangling).
//!
//! The C `beta` tensor is a `sunrealtype***` built from three separate
//! allocations so it can be indexed `beta[i][j][k]`; the port stores the same
//! logical `[sequential_methods][stages + 1][partitions]` tensor as nested
//! `Vec`s. The only place C's flat layout is observable is
//! `SplittingStepCoefficients_Create`, whose `beta_1d` argument is the
//! contiguous row-major buffer — the port indexes it with the exact same
//! `(i * (stages + 1) + j) * partitions + k` expression C's `memcpy` implies.
//! `SplittingStepCoefficients_ComposeStrangHelper` advances a *pointer into
//! the row array* in C (`return &beta[partitions - 1]`); the port passes the
//! row slice plus an explicit `beta_off` index and returns the advanced index,
//! which is the identical computation without pointer arithmetic.
//!
//! Deviations, all of the accepted classes:
//! * NULL-pointer parameter checks that the type system makes unrepresentable
//!   are dropped (`alpha`/`beta` in `_Create`, `coefficients` in `_Copy`);
//!   nullable object parameters that C reports on, or that internal callers
//!   really do pass NULL for, keep their branch as `Option<&_>`
//!   (`_Write`).
//! * `ARKODE_SplittingCoefficientsID` is a `i32` type alias plus `const`s
//!   rather than a Rust `enum`: the C enumeration has duplicate values
//!   (`ARKODE_MIN_SPLITTING_NUM == ARKODE_SPLITTING_LIE_TROTTER_1_1_2`,
//!   `ARKODE_MAX_SPLITTING_NUM == ARKODE_SPLITTING_YOSHIDA_8_6_2`), which a
//!   Rust `enum` cannot express. This is the treatment the contract already
//!   gives `ARK_INTERP_*` / `ARK_ADAPT_*`.
//! * The `.def` bodies that call `SplittingStepCoefficients_Alloc` without a
//!   NULL check dereference NULL on failure in C; the port panics
//!   (deviation class 5).

use std::cell::RefCell;
use std::rc::Rc;

use crate::arkode_impl::*;
use sundials_core::sundials_math::{SUNIpowerI, SUNRpowerR, SUNRsqrt};
use sundials_core::sundials_types::*;
use sundials_core::sundials_utils::{sun_format_e, SUNFile};

/*---------------------------------------------------------------
  Types : struct SplittingStepCoefficientsMem, SplittingStepCoefficients
  (include/arkode/arkode_splittingstep.h)
  ---------------------------------------------------------------*/

pub struct SplittingStepCoefficientsMem {
    /// weights for sum over sequential splitting methods (len ==
    /// `sequential_methods`)
    pub alpha: Vec<sunrealtype>,
    /// subintegration nodes, indexed by the sequential method, stage, and
    /// partition: `beta[i][j][k]` with `i < sequential_methods`,
    /// `j <= stages`, `k < partitions`
    pub beta: Vec<Vec<Vec<sunrealtype>>>,
    /// number of sequential splitting methods
    pub sequential_methods: i32,
    /// number of stages within each sequential splitting method
    pub stages: i32,
    /// number of RHS partitions
    pub partitions: i32,
    /// order of convergence
    pub order: i32,
}

pub type SplittingStepCoefficients = Rc<RefCell<SplittingStepCoefficientsMem>>;

/* Splitting names use the convention
 * ARKODE_SPLITTING_<name>_<stages>_<order>_<partitions> */

/// C `enum ARKODE_SplittingCoefficientsID` (see the module docs for why this
/// is an `i32` alias rather than a Rust `enum`).
pub type ARKODE_SplittingCoefficientsID = i32;

pub const ARKODE_SPLITTING_NONE: ARKODE_SplittingCoefficientsID = -1;
pub const ARKODE_SPLITTING_LIE_TROTTER_1_1_2: ARKODE_SplittingCoefficientsID = 0;
pub const ARKODE_MIN_SPLITTING_NUM: ARKODE_SplittingCoefficientsID = 0;
pub const ARKODE_SPLITTING_STRANG_2_2_2: ARKODE_SplittingCoefficientsID = 1;
pub const ARKODE_SPLITTING_BEST_2_2_2: ARKODE_SplittingCoefficientsID = 2;
pub const ARKODE_SPLITTING_SUZUKI_3_3_2: ARKODE_SplittingCoefficientsID = 3;
pub const ARKODE_SPLITTING_RUTH_3_3_2: ARKODE_SplittingCoefficientsID = 4;
pub const ARKODE_SPLITTING_YOSHIDA_4_4_2: ARKODE_SplittingCoefficientsID = 5;
pub const ARKODE_SPLITTING_YOSHIDA_8_6_2: ARKODE_SplittingCoefficientsID = 6;
pub const ARKODE_MAX_SPLITTING_NUM: ARKODE_SplittingCoefficientsID =
    ARKODE_SPLITTING_YOSHIDA_8_6_2;

/*---------------------------------------------------------------
  Routine to allocate splitting coefficients with zero values for
  alpha and beta
  ---------------------------------------------------------------*/
pub fn SplittingStepCoefficients_Alloc(
    sequential_methods: i32,
    stages: i32,
    partitions: i32,
) -> Option<SplittingStepCoefficients> {
    if sequential_methods < 1 || stages < 1 || partitions < 1 {
        return None;
    }

    /* C allocates the struct, then `calloc`s alpha, then builds the three
       levels of the beta tensor (array of row-pointer blocks, matrix of row
       pointers, contiguous `calloc`ed storage). All allocation-failure paths
       call `SplittingStepCoefficients_Destroy` and return NULL; allocation
       cannot fail here. */
    let coefficients = SplittingStepCoefficientsMem {
        alpha: vec![ZERO; sequential_methods as usize],
        beta: vec![
            vec![vec![ZERO; partitions as usize]; (stages + 1) as usize];
            sequential_methods as usize
        ],
        sequential_methods,
        stages,
        partitions,
        order: 0,
    };

    Some(Rc::new(RefCell::new(coefficients)))
}

/*---------------------------------------------------------------
  Routine to create splitting coefficients which performs a copy
  of the alpha and beta parameters
  ---------------------------------------------------------------*/
pub fn SplittingStepCoefficients_Create(
    sequential_methods: i32,
    stages: i32,
    partitions: i32,
    order: i32,
    alpha_1d: &[sunrealtype],
    beta_1d: &[sunrealtype],
) -> Option<SplittingStepCoefficients> {
    /* C: `alpha == NULL || beta == NULL` -- handled by the type system */
    if order < 1 {
        return None;
    }

    let coefficients = SplittingStepCoefficients_Alloc(sequential_methods, stages, partitions)?;

    {
        let mut c = coefficients.borrow_mut();
        c.order = order;
        /* memcpy(c->alpha, alpha, sequential_methods * sizeof(sunrealtype)) */
        for i in 0..sequential_methods as usize {
            c.alpha[i] = alpha_1d[i];
        }
        /* memcpy(c->beta[0][0], beta,
                  sequential_methods * (stages + 1) * partitions * sizeof(sunrealtype)) --
           `beta[0][0]` is the head of the contiguous tensor, laid out with
           the stride C's pointer setup implies. */
        let sm = sequential_methods as usize;
        let st = stages as usize;
        let np = partitions as usize;
        for i in 0..sm {
            for j in 0..=st {
                for k in 0..np {
                    c.beta[i][j][k] = beta_1d[(i * (st + 1) + j) * np + k];
                }
            }
        }
    }

    Some(coefficients)
}

/*---------------------------------------------------------------
  Routine to free splitting coefficients
  ---------------------------------------------------------------*/
pub fn SplittingStepCoefficients_Destroy(coefficients: &mut Option<SplittingStepCoefficients>) {
    /* C: `coefficients == NULL || *coefficients == NULL` -> return */
    if coefficients.is_none() {
        return;
    }

    /* C frees alpha, the three beta allocations, and the struct; dropping the
       `Rc` does all of it (storage lives while other clones do). */
    *coefficients = None;
}

/*---------------------------------------------------------------
  Routine to create a copy of splitting coefficients
  ---------------------------------------------------------------*/
pub fn SplittingStepCoefficients_Copy(
    coefficients: &SplittingStepCoefficients,
) -> Option<SplittingStepCoefficients> {
    /* C: `coefficients == NULL` -- handled by the type system */
    let c = coefficients.borrow();

    let coefficientsCopy =
        SplittingStepCoefficients_Alloc(c.sequential_methods, c.stages, c.partitions)?;

    {
        let mut cc = coefficientsCopy.borrow_mut();
        cc.order = c.order;
        for i in 0..c.sequential_methods as usize {
            cc.alpha[i] = c.alpha[i];
        }

        /* beta[0][0] points to the contiguous memory allocation, so C can copy
           it with a single memcpy */
        for i in 0..c.sequential_methods as usize {
            for j in 0..=c.stages as usize {
                for k in 0..c.partitions as usize {
                    cc.beta[i][j][k] = c.beta[i][j][k];
                }
            }
        }
    }

    Some(coefficientsCopy)
}

/*---------------------------------------------------------------
  Routine to load coefficients from an ID

  `arkode_splittingstep_coefficients.def` X-macro bodies, in file order.
  ---------------------------------------------------------------*/
pub fn SplittingStepCoefficients_LoadCoefficients(
    method: ARKODE_SplittingCoefficientsID,
) -> Option<SplittingStepCoefficients> {
    match method {
        ARKODE_SPLITTING_NONE => None,

        ARKODE_SPLITTING_LIE_TROTTER_1_1_2 => SplittingStepCoefficients_LieTrotter(2),

        ARKODE_SPLITTING_STRANG_2_2_2 => SplittingStepCoefficients_Strang(2),

        ARKODE_SPLITTING_BEST_2_2_2 => Some(splittingCoefficients_Best_2_2_2()),

        ARKODE_SPLITTING_SUZUKI_3_3_2 => SplittingStepCoefficients_ThirdOrderSuzuki(2),

        ARKODE_SPLITTING_RUTH_3_3_2 => Some(splittingCoefficients_Ruth_3_3_2()),

        ARKODE_SPLITTING_YOSHIDA_4_4_2 => SplittingStepCoefficients_TripleJump(2, 4),

        ARKODE_SPLITTING_YOSHIDA_8_6_2 => SplittingStepCoefficients_TripleJump(2, 6),

        _ => {
            arkProcessError(
                None,
                ARK_ILL_INPUT,
                line!() as i32,
                "SplittingStepCoefficients_LoadCoefficients",
                file!(),
                "Unknown splitting coefficients",
            );
            None
        }
    }
}

/*---------------------------------------------------------------
  Routine to load coefficients using a string representation of
  an enum entry in ARKODE_SplittingCoefficientsID
  ---------------------------------------------------------------*/
pub fn SplittingStepCoefficients_LoadCoefficientsByName(
    method: &str,
) -> Option<SplittingStepCoefficients> {
    if "ARKODE_SPLITTING_NONE" == method {
        return None;
    }
    if "ARKODE_SPLITTING_LIE_TROTTER_1_1_2" == method {
        return SplittingStepCoefficients_LieTrotter(2);
    }
    if "ARKODE_SPLITTING_STRANG_2_2_2" == method {
        return SplittingStepCoefficients_Strang(2);
    }
    if "ARKODE_SPLITTING_BEST_2_2_2" == method {
        return Some(splittingCoefficients_Best_2_2_2());
    }
    if "ARKODE_SPLITTING_SUZUKI_3_3_2" == method {
        return SplittingStepCoefficients_ThirdOrderSuzuki(2);
    }
    if "ARKODE_SPLITTING_RUTH_3_3_2" == method {
        return Some(splittingCoefficients_Ruth_3_3_2());
    }
    if "ARKODE_SPLITTING_YOSHIDA_4_4_2" == method {
        return SplittingStepCoefficients_TripleJump(2, 4);
    }
    if "ARKODE_SPLITTING_YOSHIDA_8_6_2" == method {
        return SplittingStepCoefficients_TripleJump(2, 6);
    }

    arkProcessError(
        None,
        ARK_ILL_INPUT,
        line!() as i32,
        "SplittingStepCoefficients_LoadCoefficientsByName",
        file!(),
        "Unknown splitting coefficients",
    );

    None
}

/*---------------------------------------------------------------
  Routine to convert a coefficient enum value to its string
  representation
  ---------------------------------------------------------------*/
pub fn SplittingStepCoefficients_IDToName(
    id: ARKODE_SplittingCoefficientsID,
) -> Option<&'static str> {
    /* Use X-macro to test each coefficient name */
    match id {
        ARKODE_SPLITTING_NONE => Some("ARKODE_SPLITTING_NONE"),
        ARKODE_SPLITTING_LIE_TROTTER_1_1_2 => Some("ARKODE_SPLITTING_LIE_TROTTER_1_1_2"),
        ARKODE_SPLITTING_STRANG_2_2_2 => Some("ARKODE_SPLITTING_STRANG_2_2_2"),
        ARKODE_SPLITTING_BEST_2_2_2 => Some("ARKODE_SPLITTING_BEST_2_2_2"),
        ARKODE_SPLITTING_SUZUKI_3_3_2 => Some("ARKODE_SPLITTING_SUZUKI_3_3_2"),
        ARKODE_SPLITTING_RUTH_3_3_2 => Some("ARKODE_SPLITTING_RUTH_3_3_2"),
        ARKODE_SPLITTING_YOSHIDA_4_4_2 => Some("ARKODE_SPLITTING_YOSHIDA_4_4_2"),
        ARKODE_SPLITTING_YOSHIDA_8_6_2 => Some("ARKODE_SPLITTING_YOSHIDA_8_6_2"),

        _ => {
            arkProcessError(
                None,
                ARK_ILL_INPUT,
                line!() as i32,
                "SplittingStepCoefficients_IDToName",
                file!(),
                "Unknown splitting coefficients",
            );
            None
        }
    }
}

/*---------------------------------------------------------------
  `ARKODE_SPLITTING_BEST_2_2_2` X-macro body. C does not NULL-check the
  `_Alloc` result and dereferences it directly (deviation class 5).
  ---------------------------------------------------------------*/
fn splittingCoefficients_Best_2_2_2() -> SplittingStepCoefficients {
    let coefficients = SplittingStepCoefficients_Alloc(1, 2, 2)
        .expect("SplittingStepCoefficients_Alloc(1, 2, 2)");
    {
        let mut c = coefficients.borrow_mut();
        c.order = 2;
        c.alpha[0] = 1.0;
        c.beta[0][1][0] = 1.0 - SUNRsqrt(0.5);
        c.beta[0][1][1] = SUNRsqrt(0.5);
        c.beta[0][2][0] = 1.0;
        c.beta[0][2][1] = 1.0;
    }
    coefficients
}

/*---------------------------------------------------------------
  `ARKODE_SPLITTING_RUTH_3_3_2` X-macro body.
  ---------------------------------------------------------------*/
fn splittingCoefficients_Ruth_3_3_2() -> SplittingStepCoefficients {
    let coefficients = SplittingStepCoefficients_Alloc(1, 3, 2)
        .expect("SplittingStepCoefficients_Alloc(1, 3, 2)");
    {
        let mut c = coefficients.borrow_mut();
        c.order = 3;
        c.alpha[0] = 1.0;
        c.beta[0][1][0] = 1.0;
        c.beta[0][1][1] = -1.0 / 24.0;
        c.beta[0][2][0] = 1.0 / 3.0;
        c.beta[0][2][1] = 17.0 / 24.0;
        c.beta[0][3][0] = 1.0;
        c.beta[0][3][1] = 1.0;
    }
    coefficients
}

/*---------------------------------------------------------------
  Routine to construct the standard Lie-Trotter splitting
  ---------------------------------------------------------------*/
pub fn SplittingStepCoefficients_LieTrotter(
    partitions: i32,
) -> Option<SplittingStepCoefficients> {
    let coefficients = SplittingStepCoefficients_Alloc(1, 1, partitions)?;

    {
        let mut c = coefficients.borrow_mut();
        c.order = 1;
        c.alpha[0] = 1.0;
        for i in 0..partitions as usize {
            c.beta[0][1][i] = 1.0;
        }
    }

    Some(coefficients)
}

/*---------------------------------------------------------------
  Routine to construct the standard Stang splitting
  ---------------------------------------------------------------*/
pub fn SplittingStepCoefficients_Strang(partitions: i32) -> Option<SplittingStepCoefficients> {
    SplittingStepCoefficients_TripleJump(partitions, 2)
}

/*---------------------------------------------------------------
  Routine to construct a parallel splitting method
  Phi_1(h) + Phi_2(h) + ... + Phi_p(h) - (p - 1) * y_n
  where Phi_i is the flow of partition i and p = partitions.
  ---------------------------------------------------------------*/
pub fn SplittingStepCoefficients_Parallel(partitions: i32) -> Option<SplittingStepCoefficients> {
    let coefficients = SplittingStepCoefficients_Alloc(partitions + 1, 1, partitions)?;

    {
        let mut c = coefficients.borrow_mut();
        c.order = 1;
        for i in 0..partitions as usize {
            c.alpha[i] = 1.0;
            c.beta[i][1][i] = 1.0;
        }

        c.alpha[partitions as usize] = (1 - partitions) as sunrealtype;
    }

    Some(coefficients)
}

/*---------------------------------------------------------------
  Routine to construct a symmetric parallel splitting which is
  the average of the Lie-Trotter method and its adjoint
  ---------------------------------------------------------------*/
pub fn SplittingStepCoefficients_SymmetricParallel(
    partitions: i32,
) -> Option<SplittingStepCoefficients> {
    let coefficients = SplittingStepCoefficients_Alloc(2, partitions, partitions)?;

    {
        let mut c = coefficients.borrow_mut();
        c.order = 2;
        c.alpha[0] = 0.5;
        c.alpha[1] = 0.5;

        for i in 0..partitions {
            c.beta[0][partitions as usize][i as usize] = 1.0;
            for j in (partitions - i - 1)..partitions {
                c.beta[1][(i + 1) as usize][j as usize] = 1.0;
            }
        }
    }

    Some(coefficients)
}

/*---------------------------------------------------------------
  Routine to construct a 3rd order method of Suzuki of the form
  L(p1 h) * L*(p2 h) * L(p3 h) * L*(p4 h) * L(p5 h)
  where L is a Lie-Trotter splitting and L* is its adjoint.
  Composition is denoted by *.
  ---------------------------------------------------------------*/
pub fn SplittingStepCoefficients_ThirdOrderSuzuki(
    partitions: i32,
) -> Option<SplittingStepCoefficients> {
    let coefficients = SplittingStepCoefficients_Alloc(1, 2 * partitions - 1, partitions)?;

    {
        let mut c = coefficients.borrow_mut();
        c.order = 3;
        c.alpha[0] = 1.0;

        for i in 1..partitions {
            for j in 0..partitions {
                // Constants from https://doi.org/10.1143/JPSJ.61.3015 pg. 3019
                let p1: sunrealtype = 0.2683300957817599249569552299254991394812;
                let p2: sunrealtype = 0.6513314272356399320939424082278836500821;

                c.beta[0][i as usize][j as usize] = if i + j < partitions { p1 } else { p1 + p2 };
                c.beta[0][(partitions + i - 1) as usize][j as usize] =
                    1.0 - if i + j < partitions { p1 + p2 } else { p1 };
            }
        }

        for i in 0..partitions as usize {
            c.beta[0][(2 * partitions - 1) as usize][i] = 1.0;
        }
    }

    Some(coefficients)
}

/*---------------------------------------------------------------
  Routine to construct a composition method of the form
  S(gamma_0 h)^c * S(gamma_1 h) * S(gamma_0)^c
  where S is a lower order splitting (with Stang as the base case),
  * and ^ denote composition, and c = composition_stages. This
  covers both the triple jump (c=1) and Suzuki fractal (c=2).

  C threads a `sunrealtype* const* beta` cursor through the recursion and
  returns `&beta[partitions - 1]`; the port passes the row slice untouched
  plus the current index `beta_off` and returns the advanced index.
  ---------------------------------------------------------------*/
fn SplittingStepCoefficients_ComposeStrangHelper(
    partitions: i32,
    order: i32,
    composition_stages: i32,
    start: sunrealtype,
    end: sunrealtype,
    beta: &mut [Vec<sunrealtype>],
    beta_off: usize,
) -> usize {
    let diff = end - start;
    if order == 2 {
        /* The base case is an order 2 Strang splitting */
        let mid = start + diff / 2.0;
        for j in 1..=partitions {
            for k in 0..partitions {
                beta[beta_off + j as usize][k as usize] =
                    if k + j < partitions { mid } else { end };
            }
        }

        return beta_off + (partitions - 1) as usize;
    }

    let mut beta_cur = beta_off;
    let mut start_cur = start;
    /* This is essentially the gamma coefficient from Geometric Numerical
     * Integration (https://doi.org/10.1007/3-540-30666-8) pg 44-45 scaled by the
     * current interval */
    let gamma = diff
        / ((composition_stages - 1) as sunrealtype
            - SUNRpowerR(
                (composition_stages - 1) as sunrealtype,
                1.0 / (order - 1) as sunrealtype,
            ));
    for i in 1..=composition_stages {
        /* To avoid roundoff issues, this ensures end_cur=1 for the last value of i*/
        let end_cur = if 2 * i < composition_stages {
            start + i as sunrealtype * gamma
        } else {
            end + (i - composition_stages) as sunrealtype * gamma
        };
        /* Recursively generate coefficients and shift beta_cur */
        beta_cur = SplittingStepCoefficients_ComposeStrangHelper(
            partitions,
            order - 2,
            composition_stages,
            start_cur,
            end_cur,
            beta,
            beta_cur,
        );
        start_cur = end_cur;
    }

    beta_cur
}

/*---------------------------------------------------------------
  Routine which does validation and setup before calling
  SplittingStepCoefficients_ComposeStrangHelper to fill in the
  beta coefficients
  ---------------------------------------------------------------*/
fn SplittingStepCoefficients_ComposeStrang(
    partitions: i32,
    order: i32,
    composition_stages: i32,
) -> Option<SplittingStepCoefficients> {
    if order < 2 || order % 2 != 0 {
        // Only even orders allowed
        return None;
    }

    let stages = 1 + (partitions - 1) * SUNIpowerI(composition_stages, order / 2 - 1);
    let coefficients = SplittingStepCoefficients_Alloc(1, stages, partitions)?;

    {
        let mut c = coefficients.borrow_mut();
        c.order = order;
        c.alpha[0] = 1.0;

        let beta0 = &mut c.beta[0];
        let _ = SplittingStepCoefficients_ComposeStrangHelper(
            partitions,
            order,
            composition_stages,
            0.0,
            1.0,
            beta0,
            0,
        );
    }

    Some(coefficients)
}

pub fn SplittingStepCoefficients_TripleJump(
    partitions: i32,
    order: i32,
) -> Option<SplittingStepCoefficients> {
    SplittingStepCoefficients_ComposeStrang(partitions, order, 3)
}

pub fn SplittingStepCoefficients_SuzukiFractal(
    partitions: i32,
    order: i32,
) -> Option<SplittingStepCoefficients> {
    SplittingStepCoefficients_ComposeStrang(partitions, order, 5)
}

/*---------------------------------------------------------------
  Routine to print a splitting coefficient structure
  ---------------------------------------------------------------*/
pub fn SplittingStepCoefficients_Write(
    coefficients: Option<&SplittingStepCoefficients>,
    outfile: &SUNFile,
) {
    /* C additionally rejects `alpha == NULL`, `beta == NULL`,
       `beta[0] == NULL` and `beta[0][0] == NULL`; `_Alloc` is the only
       constructor and always populates all four, so those are unrepresentable
       here. */
    if outfile.is_null() {
        return;
    }
    let coefficients = match coefficients {
        None => return,
        Some(coefficients) => coefficients,
    };

    let c = coefficients.borrow();

    outfile.write_str(&format!(
        "  sequential methods = {}\n",
        c.sequential_methods
    ));
    outfile.write_str(&format!("  stages = {}\n", c.stages));
    outfile.write_str(&format!("  partitions = {}\n", c.partitions));
    outfile.write_str(&format!("  order = {}\n", c.order));
    outfile.write_str("  alpha = ");
    for i in 0..c.sequential_methods as usize {
        outfile.write_str(&format!("{}  ", sun_format_e(c.alpha[i])));
    }
    outfile.write_str("\n");

    for i in 0..c.sequential_methods as usize {
        outfile.write_str(&format!("  beta[{i}] = \n"));
        for j in 0..=c.stages as usize {
            outfile.write_str("      ");
            for k in 0..c.partitions as usize {
                outfile.write_str(&format!("{}  ", sun_format_e(c.beta[i][j][k])));
            }
            outfile.write_str("\n");
        }
        outfile.write_str("\n");
    }
}
