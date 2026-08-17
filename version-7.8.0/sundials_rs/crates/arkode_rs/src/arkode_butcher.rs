//! Port of `src/arkode/arkode_butcher.c` (+ `include/arkode/arkode_butcher.h`
//! folded in, per the workspace module-naming rule).
//!
//! Butcher table storage plus the analytical order-condition checking
//! machinery (necessary conditions through order 6, then Butcher's
//! simplifying assumptions for higher orders).
//!
//! Representation (frozen in the crate seam spec, section 5):
//!
//! ```text
//! C                                  Rust
//! ---------------------------------  ----------------------------------
//! struct ARKodeButcherTableMem       ARKodeButcherTableMem
//! sunrealtype** A                    Vec<Vec<sunrealtype>>  (row-major)
//! sunrealtype*  c / b                Vec<sunrealtype>
//! sunrealtype*  d  (NULL = none)     Vec<sunrealtype>  (EMPTY = C NULL)
//! ARKodeButcherTable (pointer)       Rc<RefCell<ARKodeButcherTableMem>>
//! ```
//!
//! Consequences of the handle model, all of which the ARKStep / ERKStep /
//! SPRKStep ports depend on:
//!
//!  * every C `ARKodeButcherTable` argument that may legitimately be NULL
//!    is `Option<&ARKodeButcherTable>` here (stepper content fields are
//!    `Option<ARKodeButcherTable>`, so the call site is `X.as_ref()`);
//!  * `B->d != NULL` is `!B.borrow().d.is_empty()`;
//!  * `ARKodeButcherTable_Free` drops one handle. In C it frees the
//!    allocation outright, so a surviving alias dangles; here the table
//!    lives until the last `Rc` clone goes away. Strictly safer, and
//!    unobservable because `_Copy` deep-copies into a fresh `Rc`.
//!
//! `A`, `b`, `c` and `d` are filled by the one and only constructor
//! (`ARKodeButcherTable_Alloc`), so the C "critical contents" NULL checks
//! (`B->A == NULL`, `B->A[i] == NULL`, `B->b == NULL`, `B->c == NULL`)
//! cannot fire for any table this port can produce; each such check is
//! noted where it is dropped.
//!
//! Every `pow` goes through `sundials_core::sundials_math::SUNRpowerI`
//! (never `f64::powi`), and every float printed goes through
//! `sun_format_e` (C `SUN_FORMAT_E` = `"% .15e"`).

use std::cell::RefCell;
use std::rc::Rc;

/* NOTE: this module deliberately does NOT `use crate::arkode_impl::*;` --
   `arkode_butcher.c` needs nothing from `arkode_impl.h` beyond the core
   SUNDIALS types/math, and an unused glob import is a hard warning. */
use sundials_core::sundials_math::{SUNMAX, SUNRabs, SUNRpowerI, SUNRsqrt};
use sundials_core::sundials_types::*;
use sundials_core::sundials_utils::{sun_format_e, SUNFile};

/* tolerance for checking order conditions
   (C `#define TOL (SUNRsqrt(SUN_UNIT_ROUNDOFF))` -- a macro, re-evaluated
   at every use site, hence a fn and not a const) */
#[inline]
fn TOL() -> sunrealtype {
    SUNRsqrt(SUN_UNIT_ROUNDOFF)
}

/*---------------------------------------------------------------
  Types : struct ARKodeButcherTableMem, ARKodeButcherTable
  (include/arkode/arkode_butcher.h)
  ---------------------------------------------------------------*/

pub struct ARKodeButcherTableMem {
    /// method order of accuracy
    pub q: i32,
    /// embedding order of accuracy
    pub p: i32,
    /// number of stages
    pub stages: i32,
    /// Butcher table coefficients, `A[i][j]`, `stages` x `stages`
    pub A: Vec<Vec<sunrealtype>>,
    /// canopy node coefficients (len == `stages`)
    pub c: Vec<sunrealtype>,
    /// root node coefficients (len == `stages`)
    pub b: Vec<sunrealtype>,
    /// embedding coefficients (len == `stages`); EMPTY == C `NULL`, i.e.
    /// the table has no embedding
    pub d: Vec<sunrealtype>,
}

pub type ARKodeButcherTable = Rc<RefCell<ARKodeButcherTableMem>>;

/*---------------------------------------------------------------
  Routine to allocate an empty Butcher table structure
  ---------------------------------------------------------------*/
pub fn ARKodeButcherTable_Alloc(
    stages: i32,
    embedded: sunbooleantype,
) -> Option<ARKodeButcherTable> {
    /* Check for legal 'stages' value */
    if stages < 1 {
        return None;
    }

    /* Allocate Butcher table structure (C mallocs the record, NULLs every
       pointer field, then callocs each field in turn; an allocation
       failure returns NULL after freeing the partial table). */
    let n = stages as usize;

    let B = ARKodeButcherTableMem {
        /* initialize order parameters */
        q: 0,
        p: 0,
        /* set stages into B structure */
        stages,
        /* allocate rows and columns of A */
        A: vec![vec![0.0; n]; n],
        c: vec![0.0; n],
        b: vec![0.0; n],
        d: if embedded { vec![0.0; n] } else { Vec::new() },
    };

    Some(Rc::new(RefCell::new(B)))
}

/*---------------------------------------------------------------
  Routine to allocate and fill a Butcher table structure
  ---------------------------------------------------------------*/
/// C `d_1d == NULL` (no embedding) is an EMPTY `d_1d` slice here.
/// `A_1d` is row-major, exactly as in C (`A[i][j] == A_1d[i * s + j]`).
pub fn ARKodeButcherTable_Create(
    s: i32,
    q: i32,
    p: i32,
    c_1d: &[sunrealtype],
    A_1d: &[sunrealtype],
    b_1d: &[sunrealtype],
    d_1d: &[sunrealtype],
) -> Option<ARKodeButcherTable> {
    /* Check for legal number of stages */
    if s < 1 {
        return None;
    }

    /* Does the table have an embedding? */
    let embedded: sunbooleantype = if !d_1d.is_empty() { SUNTRUE } else { SUNFALSE };

    /* Allocate Butcher table structure */
    let B = ARKodeButcherTable_Alloc(s, embedded)?;

    {
        let mut Bm = B.borrow_mut();

        /* set the relevant parameters */
        Bm.stages = s;
        Bm.q = q;
        Bm.p = p;

        for i in 0..s {
            Bm.c[i as usize] = c_1d[i as usize];
            Bm.b[i as usize] = b_1d[i as usize];
            for j in 0..s {
                Bm.A[i as usize][j as usize] = A_1d[(i * s + j) as usize];
            }
        }

        if embedded {
            for i in 0..s {
                Bm.d[i as usize] = d_1d[i as usize];
            }
        }
    }

    Some(B)
}

/*---------------------------------------------------------------
  Routine to copy a Butcher table structure
  ---------------------------------------------------------------*/
pub fn ARKodeButcherTable_Copy(B: Option<&ARKodeButcherTable>) -> Option<ARKodeButcherTable> {
    /* Check for legal input */
    let B = B?;
    let Bm = B.borrow();

    /* Get the number of stages */
    let s = Bm.stages;

    /* Does the table have an embedding? */
    let embedded: sunbooleantype = if !Bm.d.is_empty() { SUNTRUE } else { SUNFALSE };

    /* Allocate Butcher table structure */
    let Bcopy = ARKodeButcherTable_Alloc(s, embedded)?;

    {
        let mut Bc = Bcopy.borrow_mut();

        /* set the relevant parameters */
        Bc.stages = Bm.stages;
        Bc.q = Bm.q;
        Bc.p = Bm.p;

        /* Copy Butcher table */
        for i in 0..s {
            Bc.c[i as usize] = Bm.c[i as usize];
            Bc.b[i as usize] = Bm.b[i as usize];
            for j in 0..s {
                Bc.A[i as usize][j as usize] = Bm.A[i as usize][j as usize];
            }
        }

        if embedded {
            for i in 0..s {
                Bc.d[i as usize] = Bm.d[i as usize];
            }
        }
    }

    Some(Bcopy)
}

/*---------------------------------------------------------------
  Routine to query the Butcher table structure workspace size
  ---------------------------------------------------------------*/
pub fn ARKodeButcherTable_Space(
    B: Option<&ARKodeButcherTable>,
    liw: &mut sunindextype,
    lrw: &mut sunindextype,
) {
    /* initialize outputs and return if B is not allocated */
    *liw = 0;
    *lrw = 0;
    let B = match B {
        None => return,
        Some(B) => B,
    };
    let Bm = B.borrow();

    /* fill outputs based on B (C evaluates the products in `int`) */
    *liw = 3;
    if !Bm.d.is_empty() {
        *lrw = (Bm.stages * (Bm.stages + 3)) as sunindextype;
    } else {
        *lrw = (Bm.stages * (Bm.stages + 2)) as sunindextype;
    }
}

/*---------------------------------------------------------------
  Routine to free a Butcher table structure
  ---------------------------------------------------------------*/
/// C frees each field and then the record itself. Here the handle is an
/// `Rc`, so dropping it releases the table once the last clone is gone.
pub fn ARKodeButcherTable_Free(B: Option<ARKodeButcherTable>) {
    drop(B);
}

/*---------------------------------------------------------------
  Routine to print a Butcher table structure
  ---------------------------------------------------------------*/
pub fn ARKodeButcherTable_Write(B: Option<&ARKodeButcherTable>, outfile: &SUNFile) {
    /* check for valid table */
    let B = match B {
        None => return,
        Some(B) => B,
    };
    let Bm = B.borrow();
    /* the C `B->A == NULL`, `B->A[i] == NULL`, `B->c == NULL` and
       `B->b == NULL` bail-outs cannot fire: `_Alloc` fills every field */

    outfile.write_str("  A = \n");
    for i in 0..Bm.stages {
        outfile.write_str("      ");
        for j in 0..Bm.stages {
            outfile.write_str(&format!(
                "{}  ",
                sun_format_e(Bm.A[i as usize][j as usize])
            ));
        }
        outfile.write_str("\n");
    }

    outfile.write_str("  c = ");
    for i in 0..Bm.stages {
        outfile.write_str(&format!("{}  ", sun_format_e(Bm.c[i as usize])));
    }
    outfile.write_str("\n");

    outfile.write_str("  b = ");
    for i in 0..Bm.stages {
        outfile.write_str(&format!("{}  ", sun_format_e(Bm.b[i as usize])));
    }
    outfile.write_str("\n");

    if !Bm.d.is_empty() {
        outfile.write_str("  d = ");
        for i in 0..Bm.stages {
            outfile.write_str(&format!("{}  ", sun_format_e(Bm.d[i as usize])));
        }
        outfile.write_str("\n");
    }
}

pub fn ARKodeButcherTable_IsStifflyAccurate(B: Option<&ARKodeButcherTable>) -> sunbooleantype {
    /* C dereferences B unconditionally here; a NULL table is UB, which
       this port maps to a deterministic panic (deviation class 5). */
    let B = B.expect("ARKodeButcherTable_IsStifflyAccurate: B = NULL");
    let Bm = B.borrow();
    for i in 0..Bm.stages {
        if SUNRabs(Bm.b[i as usize] - Bm.A[(Bm.stages - 1) as usize][i as usize])
            > 100.0 * SUN_UNIT_ROUNDOFF
        {
            return SUNFALSE;
        }
    }
    SUNTRUE
}

/*---------------------------------------------------------------
  Routine to determine the analytical order of accuracy for a
  specified Butcher table.  We check the analytical [necessary]
  order conditions up through order 6.  After that, we revert to
  the [sufficient] Butcher simplifying assumptions.

  Inputs:
     B: Butcher table to check
     outfile: file pointer to print results; if NULL then no
        outputs are printed

  Outputs:
     q: measured order of accuracy for method
     p: measured order of accuracy for embedding [0 if not present]

  Return values:
     0 (success): internal {q,p} values match analytical order
     1 (warning): internal {q,p} values are lower than analytical
        order, or method achieves maximum order possible with this
        routine and internal {q,p} are higher.
    -1 (failure): internal p and q values are higher than analytical
         order
    -2 (failure): NULL-valued B (or critical contents)

  Note: for embedded methods, if the return flags for p and q would
  differ, failure takes precedence over warning, which takes
  precedence over success.
  ---------------------------------------------------------------*/
pub fn ARKodeButcherTable_CheckOrder(
    B: Option<&ARKodeButcherTable>,
    q: &mut i32,
    p: &mut i32,
    outfile: &SUNFile,
) -> i32 {
    /* local variables */
    let mut alltrue: sunbooleantype;
    *q = 0;
    *p = 0;

    /* verify non-NULL Butcher table structure and contents */
    let B = match B {
        None => return -2,
        Some(B) => B,
    };
    let Bm = B.borrow();
    if Bm.stages < 1 {
        return -2;
    }
    /* the C `B->A == NULL`, `B->A[i] == NULL`, `B->c == NULL` and
       `B->b == NULL` bail-outs cannot fire: `_Alloc` fills every field */

    /* set shortcuts for Butcher table components */
    let A: &[Vec<sunrealtype>] = &Bm.A;
    let mut b: &[sunrealtype] = &Bm.b;
    let c: &[sunrealtype] = &Bm.c;
    let d: &[sunrealtype] = &Bm.d;
    let s: i32 = Bm.stages;

    /* check method order */
    if !outfile.is_null() {
        outfile.write_str("ARKodeButcherTable_CheckOrder:\n");
    }

    /*    row sum condition */
    if arkode_butcher_rowsum(A, c, s) {
        *q = 0;
    } else {
        *q = -1;
        if !outfile.is_null() {
            outfile.write_str("  method fails row sum condition\n");
        }
    }
    /*    order 1 condition */
    if *q == 0 {
        if arkode_butcher_order1(b, s) {
            *q = 1;
        } else if !outfile.is_null() {
            outfile.write_str("  method fails order 1 condition\n");
        }
    }
    /*    order 2 condition */
    if *q == 1 {
        if arkode_butcher_order2(b, c, s) {
            *q = 2;
        } else if !outfile.is_null() {
            outfile.write_str("  method fails order 2 condition\n");
        }
    }
    /*    order 3 conditions */
    if *q == 2 {
        alltrue = SUNTRUE;
        if !arkode_butcher_order3a(b, c, c, s) {
            alltrue = SUNFALSE;
            if !outfile.is_null() {
                outfile.write_str("  method fails order 3 condition A\n");
            }
        }
        if !arkode_butcher_order3b(b, A, c, s) {
            alltrue = SUNFALSE;
            if !outfile.is_null() {
                outfile.write_str("  method fails order 3 condition B\n");
            }
        }
        if alltrue {
            *q = 3;
        }
    }
    /*    order 4 conditions */
    if *q == 3 {
        alltrue = SUNTRUE;
        if !arkode_butcher_order4a(b, c, c, c, s) {
            alltrue = SUNFALSE;
            if !outfile.is_null() {
                outfile.write_str("  method fails order 4 condition A\n");
            }
        }
        if !arkode_butcher_order4b(b, c, A, c, s) {
            alltrue = SUNFALSE;
            if !outfile.is_null() {
                outfile.write_str("  method fails order 4 condition B\n");
            }
        }
        if !arkode_butcher_order4c(b, A, c, c, s) {
            alltrue = SUNFALSE;
            if !outfile.is_null() {
                outfile.write_str("  method fails order 4 condition C\n");
            }
        }
        if !arkode_butcher_order4d(b, A, A, c, s) {
            alltrue = SUNFALSE;
            if !outfile.is_null() {
                outfile.write_str("  method fails order 4 condition D\n");
            }
        }
        if alltrue {
            *q = 4;
        }
    }
    /*    order 5 conditions */
    if *q == 4 {
        alltrue = SUNTRUE;
        if !arkode_butcher_order5a(b, c, c, c, c, s) {
            alltrue = SUNFALSE;
            if !outfile.is_null() {
                outfile.write_str("  method fails order 5 condition A\n");
            }
        }
        if !arkode_butcher_order5b(b, c, c, A, c, s) {
            alltrue = SUNFALSE;
            if !outfile.is_null() {
                outfile.write_str("  method fails order 5 condition B\n");
            }
        }
        if !arkode_butcher_order5c(b, A, c, A, c, s) {
            alltrue = SUNFALSE;
            if !outfile.is_null() {
                outfile.write_str("  method fails order 5 condition C\n");
            }
        }
        if !arkode_butcher_order5d(b, c, A, c, c, s) {
            alltrue = SUNFALSE;
            if !outfile.is_null() {
                outfile.write_str("  method fails order 5 condition D\n");
            }
        }
        if !arkode_butcher_order5e(b, A, c, c, c, s) {
            alltrue = SUNFALSE;
            if !outfile.is_null() {
                outfile.write_str("  method fails order 5 condition E\n");
            }
        }
        if !arkode_butcher_order5f(b, c, A, A, c, s) {
            alltrue = SUNFALSE;
            if !outfile.is_null() {
                outfile.write_str("  method fails order 5 condition F\n");
            }
        }
        if !arkode_butcher_order5g(b, A, c, A, c, s) {
            alltrue = SUNFALSE;
            if !outfile.is_null() {
                outfile.write_str("  method fails order 5 condition G\n");
            }
        }
        if !arkode_butcher_order5h(b, A, A, c, c, s) {
            alltrue = SUNFALSE;
            if !outfile.is_null() {
                outfile.write_str("  method fails order 5 condition H\n");
            }
        }
        if !arkode_butcher_order5i(b, A, A, A, c, s) {
            alltrue = SUNFALSE;
            if !outfile.is_null() {
                outfile.write_str("  method fails order 5 condition I\n");
            }
        }
        if alltrue {
            *q = 5;
        }
    }
    /*    order 6 conditions */
    if *q == 5 {
        alltrue = SUNTRUE;
        if !arkode_butcher_order6a(b, c, c, c, c, c, s) {
            alltrue = SUNFALSE;
            if !outfile.is_null() {
                outfile.write_str("  method fails order 6 condition A\n");
            }
        }
        if !arkode_butcher_order6b(b, c, c, c, A, c, s) {
            alltrue = SUNFALSE;
            if !outfile.is_null() {
                outfile.write_str("  method fails order 6 condition B\n");
            }
        }
        if !arkode_butcher_order6c(b, c, A, c, A, c, s) {
            alltrue = SUNFALSE;
            if !outfile.is_null() {
                outfile.write_str("  method fails order 6 condition C\n");
            }
        }
        if !arkode_butcher_order6d(b, c, c, A, c, c, s) {
            alltrue = SUNFALSE;
            if !outfile.is_null() {
                outfile.write_str("  method fails order 6 condition D\n");
            }
        }
        if !arkode_butcher_order6e(b, c, c, A, A, c, s) {
            alltrue = SUNFALSE;
            if !outfile.is_null() {
                outfile.write_str("  method fails order 6 condition E\n");
            }
        }
        if !arkode_butcher_order6f(b, A, A, c, A, c, s) {
            alltrue = SUNFALSE;
            if !outfile.is_null() {
                outfile.write_str("  method fails order 6 condition F\n");
            }
        }
        if !arkode_butcher_order6g(b, c, A, c, c, c, s) {
            alltrue = SUNFALSE;
            if !outfile.is_null() {
                outfile.write_str("  method fails order 6 condition G\n");
            }
        }
        if !arkode_butcher_order6h(b, c, A, c, A, c, s) {
            alltrue = SUNFALSE;
            if !outfile.is_null() {
                outfile.write_str("  method fails order 6 condition H\n");
            }
        }
        if !arkode_butcher_order6i(b, c, A, A, c, c, s) {
            alltrue = SUNFALSE;
            if !outfile.is_null() {
                outfile.write_str("  method fails order 6 condition I\n");
            }
        }
        if !arkode_butcher_order6j(b, c, A, A, A, c, s) {
            alltrue = SUNFALSE;
            if !outfile.is_null() {
                outfile.write_str("  method fails order 6 condition J\n");
            }
        }
        if !arkode_butcher_order6k(b, A, c, c, c, c, s) {
            alltrue = SUNFALSE;
            if !outfile.is_null() {
                outfile.write_str("  method fails order 6 condition K\n");
            }
        }
        if !arkode_butcher_order6l(b, A, c, c, A, c, s) {
            alltrue = SUNFALSE;
            if !outfile.is_null() {
                outfile.write_str("  method fails order 6 condition L\n");
            }
        }
        if !arkode_butcher_order6m(b, A, A, c, A, c, s) {
            alltrue = SUNFALSE;
            if !outfile.is_null() {
                outfile.write_str("  method fails order 6 condition M\n");
            }
        }
        if !arkode_butcher_order6n(b, A, c, A, c, c, s) {
            alltrue = SUNFALSE;
            if !outfile.is_null() {
                outfile.write_str("  method fails order 6 condition N\n");
            }
        }
        if !arkode_butcher_order6o(b, A, c, A, A, c, s) {
            alltrue = SUNFALSE;
            if !outfile.is_null() {
                outfile.write_str("  method fails order 6 condition O\n");
            }
        }
        if !arkode_butcher_order6p(b, A, A, c, c, c, s) {
            alltrue = SUNFALSE;
            if !outfile.is_null() {
                outfile.write_str("  method fails order 6 condition P\n");
            }
        }
        if !arkode_butcher_order6q(b, A, A, c, A, c, s) {
            alltrue = SUNFALSE;
            if !outfile.is_null() {
                outfile.write_str("  method fails order 6 condition Q\n");
            }
        }
        if !arkode_butcher_order6r(b, A, A, A, c, c, s) {
            alltrue = SUNFALSE;
            if !outfile.is_null() {
                outfile.write_str("  method fails order 6 condition R\n");
            }
        }
        if !arkode_butcher_order6s(b, A, A, A, A, c, s) {
            alltrue = SUNFALSE;
            if !outfile.is_null() {
                outfile.write_str("  method fails order 6 condition S\n");
            }
        }
        if alltrue {
            *q = 6;
        }
    }
    /*    higher order conditions (via simplifying assumptions) */
    if *q == 6 {
        if !outfile.is_null() {
            outfile.write_str("  method order >= 6; reverting to simplifying assumptions\n");
        }
        let q_SA = __ButcherSimplifyingAssumptions(A, b, c, s);
        *q = SUNMAX(*q, q_SA);
        if !outfile.is_null() {
            outfile.write_str(&format!("  method order = {}\n", *q));
        }
    }

    /* check embedding order */
    if !d.is_empty() {
        if !outfile.is_null() {
            outfile.write_str("\n");
        }
        b = d;

        /*    row sum condition */
        if arkode_butcher_rowsum(A, c, s) {
            *p = 0;
        } else {
            *p = -1;
            if !outfile.is_null() {
                outfile.write_str("  embedding fails row sum condition\n");
            }
        }
        /*    order 1 condition */
        if *p == 0 {
            if arkode_butcher_order1(b, s) {
                *p = 1;
            } else if !outfile.is_null() {
                outfile.write_str("  embedding fails order 1 condition\n");
            }
        }
        /*    order 2 condition */
        if *p == 1 {
            if arkode_butcher_order2(b, c, s) {
                *p = 2;
            } else if !outfile.is_null() {
                outfile.write_str("  embedding fails order 2 condition\n");
            }
        }
        /*    order 3 conditions */
        if *p == 2 {
            alltrue = SUNTRUE;
            if !arkode_butcher_order3a(b, c, c, s) {
                alltrue = SUNFALSE;
                if !outfile.is_null() {
                    outfile.write_str("  embedding fails order 3 condition A\n");
                }
            }
            if !arkode_butcher_order3b(b, A, c, s) {
                alltrue = SUNFALSE;
                if !outfile.is_null() {
                    outfile.write_str("  embedding fails order 3 condition B\n");
                }
            }
            if alltrue {
                *p = 3;
            }
        }
        /*    order 4 conditions */
        if *p == 3 {
            alltrue = SUNTRUE;
            if !arkode_butcher_order4a(b, c, c, c, s) {
                alltrue = SUNFALSE;
                if !outfile.is_null() {
                    outfile.write_str("  embedding fails order 4 condition A\n");
                }
            }
            if !arkode_butcher_order4b(b, c, A, c, s) {
                alltrue = SUNFALSE;
                if !outfile.is_null() {
                    outfile.write_str("  embedding fails order 4 condition B\n");
                }
            }
            if !arkode_butcher_order4c(b, A, c, c, s) {
                alltrue = SUNFALSE;
                if !outfile.is_null() {
                    outfile.write_str("  embedding fails order 4 condition C\n");
                }
            }
            if !arkode_butcher_order4d(b, A, A, c, s) {
                alltrue = SUNFALSE;
                if !outfile.is_null() {
                    outfile.write_str("  embedding fails order 4 condition D\n");
                }
            }
            if alltrue {
                *p = 4;
            }
        }
        /*    order 5 conditions */
        if *p == 4 {
            alltrue = SUNTRUE;
            if !arkode_butcher_order5a(b, c, c, c, c, s) {
                alltrue = SUNFALSE;
                if !outfile.is_null() {
                    outfile.write_str("  embedding fails order 5 condition A\n");
                }
            }
            if !arkode_butcher_order5b(b, c, c, A, c, s) {
                alltrue = SUNFALSE;
                if !outfile.is_null() {
                    outfile.write_str("  embedding fails order 5 condition B\n");
                }
            }
            if !arkode_butcher_order5c(b, A, c, A, c, s) {
                alltrue = SUNFALSE;
                if !outfile.is_null() {
                    outfile.write_str("  embedding fails order 5 condition C\n");
                }
            }
            if !arkode_butcher_order5d(b, c, A, c, c, s) {
                alltrue = SUNFALSE;
                if !outfile.is_null() {
                    outfile.write_str("  embedding fails order 5 condition D\n");
                }
            }
            if !arkode_butcher_order5e(b, A, c, c, c, s) {
                alltrue = SUNFALSE;
                if !outfile.is_null() {
                    outfile.write_str("  embedding fails order 5 condition E\n");
                }
            }
            if !arkode_butcher_order5f(b, c, A, A, c, s) {
                alltrue = SUNFALSE;
                if !outfile.is_null() {
                    outfile.write_str("  embedding fails order 5 condition F\n");
                }
            }
            if !arkode_butcher_order5g(b, A, c, A, c, s) {
                alltrue = SUNFALSE;
                if !outfile.is_null() {
                    outfile.write_str("  embedding fails order 5 condition G\n");
                }
            }
            if !arkode_butcher_order5h(b, A, A, c, c, s) {
                alltrue = SUNFALSE;
                if !outfile.is_null() {
                    outfile.write_str("  embedding fails order 5 condition H\n");
                }
            }
            if !arkode_butcher_order5i(b, A, A, A, c, s) {
                alltrue = SUNFALSE;
                if !outfile.is_null() {
                    outfile.write_str("  embedding fails order 5 condition I\n");
                }
            }
            if alltrue {
                *p = 5;
            }
        }
        /*    order 6 conditions */
        if *p == 5 {
            alltrue = SUNTRUE;
            if !arkode_butcher_order6a(b, c, c, c, c, c, s) {
                alltrue = SUNFALSE;
                if !outfile.is_null() {
                    outfile.write_str("  embedding fails order 6 condition A\n");
                }
            }
            if !arkode_butcher_order6b(b, c, c, c, A, c, s) {
                alltrue = SUNFALSE;
                if !outfile.is_null() {
                    outfile.write_str("  embedding fails order 6 condition B\n");
                }
            }
            if !arkode_butcher_order6c(b, c, A, c, A, c, s) {
                alltrue = SUNFALSE;
                if !outfile.is_null() {
                    outfile.write_str("  embedding fails order 6 condition C\n");
                }
            }
            if !arkode_butcher_order6d(b, c, c, A, c, c, s) {
                alltrue = SUNFALSE;
                if !outfile.is_null() {
                    outfile.write_str("  embedding fails order 6 condition D\n");
                }
            }
            if !arkode_butcher_order6e(b, c, c, A, A, c, s) {
                alltrue = SUNFALSE;
                if !outfile.is_null() {
                    outfile.write_str("  embedding fails order 6 condition E\n");
                }
            }
            if !arkode_butcher_order6f(b, A, A, c, A, c, s) {
                alltrue = SUNFALSE;
                if !outfile.is_null() {
                    outfile.write_str("  embedding fails order 6 condition F\n");
                }
            }
            if !arkode_butcher_order6g(b, c, A, c, c, c, s) {
                alltrue = SUNFALSE;
                if !outfile.is_null() {
                    outfile.write_str("  embedding fails order 6 condition G\n");
                }
            }
            if !arkode_butcher_order6h(b, c, A, c, A, c, s) {
                alltrue = SUNFALSE;
                if !outfile.is_null() {
                    outfile.write_str("  embedding fails order 6 condition H\n");
                }
            }
            if !arkode_butcher_order6i(b, c, A, A, c, c, s) {
                alltrue = SUNFALSE;
                if !outfile.is_null() {
                    outfile.write_str("  embedding fails order 6 condition I\n");
                }
            }
            if !arkode_butcher_order6j(b, c, A, A, A, c, s) {
                alltrue = SUNFALSE;
                if !outfile.is_null() {
                    outfile.write_str("  embedding fails order 6 condition J\n");
                }
            }
            if !arkode_butcher_order6k(b, A, c, c, c, c, s) {
                alltrue = SUNFALSE;
                if !outfile.is_null() {
                    outfile.write_str("  embedding fails order 6 condition K\n");
                }
            }
            if !arkode_butcher_order6l(b, A, c, c, A, c, s) {
                alltrue = SUNFALSE;
                if !outfile.is_null() {
                    outfile.write_str("  embedding fails order 6 condition L\n");
                }
            }
            if !arkode_butcher_order6m(b, A, A, c, A, c, s) {
                alltrue = SUNFALSE;
                if !outfile.is_null() {
                    outfile.write_str("  embedding fails order 6 condition M\n");
                }
            }
            if !arkode_butcher_order6n(b, A, c, A, c, c, s) {
                alltrue = SUNFALSE;
                if !outfile.is_null() {
                    outfile.write_str("  embedding fails order 6 condition N\n");
                }
            }
            if !arkode_butcher_order6o(b, A, c, A, A, c, s) {
                alltrue = SUNFALSE;
                if !outfile.is_null() {
                    outfile.write_str("  embedding fails order 6 condition O\n");
                }
            }
            if !arkode_butcher_order6p(b, A, A, c, c, c, s) {
                alltrue = SUNFALSE;
                if !outfile.is_null() {
                    outfile.write_str("  embedding fails order 6 condition P\n");
                }
            }
            if !arkode_butcher_order6q(b, A, A, c, A, c, s) {
                alltrue = SUNFALSE;
                if !outfile.is_null() {
                    outfile.write_str("  embedding fails order 6 condition Q\n");
                }
            }
            if !arkode_butcher_order6r(b, A, A, A, c, c, s) {
                alltrue = SUNFALSE;
                if !outfile.is_null() {
                    outfile.write_str("  embedding fails order 6 condition R\n");
                }
            }
            if !arkode_butcher_order6s(b, A, A, A, A, c, s) {
                alltrue = SUNFALSE;
                if !outfile.is_null() {
                    outfile.write_str("  embedding fails order 6 condition S\n");
                }
            }
            if alltrue {
                *p = 6;
            }
        }
        /*    higher order conditions (via simplifying assumptions) */
        if *p == 6 {
            if !outfile.is_null() {
                outfile
                    .write_str("  embedding order >= 6; reverting to simplifying assumptions\n");
            }
            let p_SA = __ButcherSimplifyingAssumptions(A, b, c, s);
            *p = SUNMAX(*p, p_SA);
            if !outfile.is_null() {
                outfile.write_str(&format!("  embedding order = {}\n", *p));
            }
        }
    }

    /* compare results against stored values and return */

    /*    check failure modes first */
    if (*q < Bm.q) && (*q < 6) {
        return -1;
    }
    if !d.is_empty() && (*p < Bm.p) && (*p < 6) {
        return -1;
    }

    /*    check warning modes */
    if *q > Bm.q {
        return 1;
    }
    if !d.is_empty() && (*p > Bm.p) {
        return 1;
    }
    if (*q < Bm.q) && (*q >= 6) {
        return 1;
    }
    if !d.is_empty() && (*p < Bm.p) && (*p >= 6) {
        return 1;
    }

    /*    return success */
    0
}

/*---------------------------------------------------------------
  Routine to determine the analytical order of accuracy for a
  specified pair of Butcher tables in an ARK pair.  We check the
  analytical order conditions up through order 6.

  Inputs:
     B1, B2: Butcher tables to check
     outfile: file pointer to print results; if NULL then no
        outputs are printed

  Outputs:
     q: measured order of accuracy for method
     p: measured order of accuracy for embedding [0 if not present]

  Return values:
     0 (success): completed checks
     1 (warning): internal {q,p} values are lower than analytical
        order, or method achieves maximum order possible with this
        routine and internal {q,p} are higher.
    -1 (failure): NULL-valued B1, B2 (or critical contents)

  Note: for embedded methods, if the return flags for p and q would
  differ, warning takes precedence over success.
  ---------------------------------------------------------------*/
pub fn ARKodeButcherTable_CheckARKOrder(
    B1: Option<&ARKodeButcherTable>,
    B2: Option<&ARKodeButcherTable>,
    q: &mut i32,
    p: &mut i32,
    outfile: &SUNFile,
) -> i32 {
    /* local variables */
    let mut alltrue: sunbooleantype;
    *q = 0;
    *p = 0;

    /* verify non-NULL Butcher table structure and contents */
    let B1 = match B1 {
        None => return -1,
        Some(B1) => B1,
    };
    let B1m = B1.borrow();
    if B1m.stages < 1 {
        return -1;
    }
    /* the C `B1->A == NULL`, `B1->A[i] == NULL`, `B1->c == NULL` and
       `B1->b == NULL` bail-outs cannot fire: `_Alloc` fills every field */
    let B2 = match B2 {
        None => return -1,
        Some(B2) => B2,
    };
    let B2m = B2.borrow();
    if B2m.stages < 1 {
        return -1;
    }
    /* likewise for the `B2->...` content checks */
    if B1m.stages != B2m.stages {
        return -1;
    }

    /* set shortcuts for Butcher table components
       (NOTE: `d[1] = B1->d` is upstream's code, transcribed verbatim) */
    let A: [&[Vec<sunrealtype>]; 2] = [&B1m.A, &B2m.A];
    let b: [&[sunrealtype]; 2] = [&B1m.b, &B2m.b];
    let c: [&[sunrealtype]; 2] = [&B1m.c, &B2m.c];
    let d: [&[sunrealtype]; 2] = [&B1m.d, &B1m.d];
    let s: i32 = B1m.stages;

    /* check method order */
    if !outfile.is_null() {
        outfile.write_str("ARKodeButcherTable_CheckARKOrder:\n");
    }

    /*    row sum conditions */
    if arkode_butcher_rowsum(A[0], c[0], s) && arkode_butcher_rowsum(A[1], c[1], s) {
        *q = 0;
    } else {
        *q = -1;
        if !outfile.is_null() {
            outfile.write_str("  method fails row sum conditions\n");
        }
    }
    /*    order 1 conditions */
    if *q == 0 {
        if arkode_butcher_order1(b[0], s) && arkode_butcher_order1(b[1], s) {
            *q = 1;
        } else if !outfile.is_null() {
            outfile.write_str("  method fails order 1 conditions\n");
        }
    }
    /*    order 2 conditions */
    if *q == 1 {
        alltrue = SUNTRUE;
        for i in 0..2 {
            for j in 0..2 {
                alltrue = alltrue && arkode_butcher_order2(b[i], c[j], s);
            }
        }
        if alltrue {
            *q = 2;
        } else if !outfile.is_null() {
            outfile.write_str("  method fails order 2 conditions\n");
        }
    }
    /*    order 3 conditions */
    if *q == 2 {
        alltrue = SUNTRUE;
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    alltrue = alltrue && arkode_butcher_order3a(b[i], c[j], c[k], s);
                }
            }
        }
        if !alltrue && !outfile.is_null() {
            outfile.write_str("  method fails order 3 conditions A\n");
        }
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    alltrue = alltrue && arkode_butcher_order3b(b[i], A[j], c[k], s);
                }
            }
        }
        if !alltrue && !outfile.is_null() {
            outfile.write_str("  method fails order 3 conditions B\n");
        }
        if alltrue {
            *q = 3;
        }
    }
    /*    order 4 conditions */
    if *q == 3 {
        alltrue = SUNTRUE;
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    for l in 0..2 {
                        alltrue =
                            alltrue && arkode_butcher_order4a(b[i], c[j], c[k], c[l], s);
                    }
                }
            }
        }
        if !alltrue && !outfile.is_null() {
            outfile.write_str("  method fails order 4 conditions A\n");
        }
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    for l in 0..2 {
                        alltrue =
                            alltrue && arkode_butcher_order4b(b[i], c[j], A[k], c[l], s);
                    }
                }
            }
        }
        if !alltrue && !outfile.is_null() {
            outfile.write_str("  method fails order 4 conditions B\n");
        }
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    for l in 0..2 {
                        alltrue =
                            alltrue && arkode_butcher_order4c(b[i], A[j], c[k], c[l], s);
                    }
                }
            }
        }
        if !alltrue && !outfile.is_null() {
            outfile.write_str("  method fails order 4 conditions C\n");
        }
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    for l in 0..2 {
                        alltrue =
                            alltrue && arkode_butcher_order4d(b[i], A[j], A[k], c[l], s);
                    }
                }
            }
        }
        if !alltrue && !outfile.is_null() {
            outfile.write_str("  method fails order 4 conditions D\n");
        }
        if alltrue {
            *q = 4;
        }
    }
    /*    order 5 conditions */
    if *q == 4 {
        alltrue = SUNTRUE;
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    for l in 0..2 {
                        for m in 0..2 {
                            alltrue = alltrue
                                && arkode_butcher_order5a(b[i], c[j], c[k], c[l], c[m], s);
                        }
                    }
                }
            }
        }
        if !alltrue && !outfile.is_null() {
            outfile.write_str("  method fails order 5 conditions A\n");
        }
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    for l in 0..2 {
                        for m in 0..2 {
                            alltrue = alltrue
                                && arkode_butcher_order5b(b[i], c[j], c[k], A[l], c[m], s);
                        }
                    }
                }
            }
        }
        if !alltrue && !outfile.is_null() {
            outfile.write_str("  method fails order 5 conditions B\n");
        }
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    for l in 0..2 {
                        for m in 0..2 {
                            alltrue = alltrue
                                && arkode_butcher_order5c(b[i], A[j], c[k], A[l], c[m], s);
                        }
                    }
                }
            }
        }
        if !alltrue && !outfile.is_null() {
            outfile.write_str("  method fails order 5 conditions C\n");
        }
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    for l in 0..2 {
                        for m in 0..2 {
                            alltrue = alltrue
                                && arkode_butcher_order5d(b[i], c[j], A[k], c[l], c[m], s);
                        }
                    }
                }
            }
        }
        if !alltrue && !outfile.is_null() {
            outfile.write_str("  method fails order 5 conditions D\n");
        }
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    for l in 0..2 {
                        for m in 0..2 {
                            alltrue = alltrue
                                && arkode_butcher_order5e(b[i], A[j], c[k], c[l], c[m], s);
                        }
                    }
                }
            }
        }
        if !alltrue && !outfile.is_null() {
            outfile.write_str("  method fails order 5 conditions E\n");
        }
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    for l in 0..2 {
                        for m in 0..2 {
                            alltrue = alltrue
                                && arkode_butcher_order5f(b[i], c[j], A[k], A[l], c[m], s);
                        }
                    }
                }
            }
        }
        if !alltrue && !outfile.is_null() {
            outfile.write_str("  method fails order 5 conditions F\n");
        }
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    for l in 0..2 {
                        for m in 0..2 {
                            alltrue = alltrue
                                && arkode_butcher_order5g(b[i], A[j], c[k], A[l], c[m], s);
                        }
                    }
                }
            }
        }
        if !alltrue && !outfile.is_null() {
            outfile.write_str("  method fails order 5 conditions G\n");
        }
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    for l in 0..2 {
                        for m in 0..2 {
                            alltrue = alltrue
                                && arkode_butcher_order5h(b[i], A[j], A[k], c[l], c[m], s);
                        }
                    }
                }
            }
        }
        if !alltrue && !outfile.is_null() {
            outfile.write_str("  method fails order 5 conditions H\n");
        }
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    for l in 0..2 {
                        for m in 0..2 {
                            alltrue = alltrue
                                && arkode_butcher_order5i(b[i], A[j], A[k], A[l], c[m], s);
                        }
                    }
                }
            }
        }
        if !alltrue && !outfile.is_null() {
            outfile.write_str("  method fails order 5 conditions I\n");
        }
        if alltrue {
            *q = 5;
        }
    }
    /*    order 6 conditions */
    if *q == 5 {
        alltrue = SUNTRUE;
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    for l in 0..2 {
                        for m in 0..2 {
                            for n in 0..2 {
                                alltrue = alltrue
                                    && arkode_butcher_order6a(
                                        b[i], c[j], c[k], c[l], c[m], c[n], s,
                                    );
                            }
                        }
                    }
                }
            }
        }
        if !alltrue && !outfile.is_null() {
            outfile.write_str("  method fails order 6 conditions A\n");
        }
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    for l in 0..2 {
                        for m in 0..2 {
                            for n in 0..2 {
                                alltrue = alltrue
                                    && arkode_butcher_order6b(
                                        b[i], c[j], c[k], c[l], A[m], c[n], s,
                                    );
                            }
                        }
                    }
                }
            }
        }
        if !alltrue && !outfile.is_null() {
            outfile.write_str("  method fails order 6 conditions B\n");
        }
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    for l in 0..2 {
                        for m in 0..2 {
                            for n in 0..2 {
                                alltrue = alltrue
                                    && arkode_butcher_order6c(
                                        b[i], c[j], A[k], c[l], A[m], c[n], s,
                                    );
                            }
                        }
                    }
                }
            }
        }
        if !alltrue && !outfile.is_null() {
            outfile.write_str("  method fails order 6 conditions C\n");
        }
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    for l in 0..2 {
                        for m in 0..2 {
                            for n in 0..2 {
                                alltrue = alltrue
                                    && arkode_butcher_order6d(
                                        b[i], c[j], c[k], A[l], c[m], c[n], s,
                                    );
                            }
                        }
                    }
                }
            }
        }
        if !alltrue && !outfile.is_null() {
            outfile.write_str("  method fails order 6 conditions D\n");
        }
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    for l in 0..2 {
                        for m in 0..2 {
                            for n in 0..2 {
                                alltrue = alltrue
                                    && arkode_butcher_order6e(
                                        b[i], c[j], c[k], A[l], A[m], c[n], s,
                                    );
                            }
                        }
                    }
                }
            }
        }
        if !alltrue && !outfile.is_null() {
            outfile.write_str("  method fails order 6 conditions E\n");
        }
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    for l in 0..2 {
                        for m in 0..2 {
                            for n in 0..2 {
                                alltrue = alltrue
                                    && arkode_butcher_order6f(
                                        b[i], A[j], A[k], c[l], A[m], c[n], s,
                                    );
                            }
                        }
                    }
                }
            }
        }
        if !alltrue && !outfile.is_null() {
            outfile.write_str("  method fails order 6 conditions F\n");
        }
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    for l in 0..2 {
                        for m in 0..2 {
                            for n in 0..2 {
                                alltrue = alltrue
                                    && arkode_butcher_order6g(
                                        b[i], c[j], A[k], c[l], c[m], c[n], s,
                                    );
                            }
                        }
                    }
                }
            }
        }
        if !alltrue && !outfile.is_null() {
            outfile.write_str("  method fails order 6 conditions G\n");
        }
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    for l in 0..2 {
                        for m in 0..2 {
                            for n in 0..2 {
                                alltrue = alltrue
                                    && arkode_butcher_order6h(
                                        b[i], c[j], A[k], c[l], A[m], c[n], s,
                                    );
                            }
                        }
                    }
                }
            }
        }
        if !alltrue && !outfile.is_null() {
            outfile.write_str("  method fails order 6 conditions H\n");
        }
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    for l in 0..2 {
                        for m in 0..2 {
                            for n in 0..2 {
                                alltrue = alltrue
                                    && arkode_butcher_order6i(
                                        b[i], c[j], A[k], A[l], c[m], c[n], s,
                                    );
                            }
                        }
                    }
                }
            }
        }
        if !alltrue && !outfile.is_null() {
            outfile.write_str("  method fails order 6 conditions I\n");
        }
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    for l in 0..2 {
                        for m in 0..2 {
                            for n in 0..2 {
                                alltrue = alltrue
                                    && arkode_butcher_order6j(
                                        b[i], c[j], A[k], A[l], A[m], c[n], s,
                                    );
                            }
                        }
                    }
                }
            }
        }
        if !alltrue && !outfile.is_null() {
            outfile.write_str("  method fails order 6 conditions J\n");
        }
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    for l in 0..2 {
                        for m in 0..2 {
                            for n in 0..2 {
                                alltrue = alltrue
                                    && arkode_butcher_order6k(
                                        b[i], A[j], c[k], c[l], c[m], c[n], s,
                                    );
                            }
                        }
                    }
                }
            }
        }
        if !alltrue && !outfile.is_null() {
            outfile.write_str("  method fails order 6 conditions K\n");
        }
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    for l in 0..2 {
                        for m in 0..2 {
                            for n in 0..2 {
                                alltrue = alltrue
                                    && arkode_butcher_order6l(
                                        b[i], A[j], c[k], c[l], A[m], c[n], s,
                                    );
                            }
                        }
                    }
                }
            }
        }
        if !alltrue && !outfile.is_null() {
            outfile.write_str("  method fails order 6 conditions L\n");
        }
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    for l in 0..2 {
                        for m in 0..2 {
                            for n in 0..2 {
                                alltrue = alltrue
                                    && arkode_butcher_order6m(
                                        b[i], A[j], A[k], c[l], A[m], c[n], s,
                                    );
                            }
                        }
                    }
                }
            }
        }
        if !alltrue && !outfile.is_null() {
            outfile.write_str("  method fails order 6 conditions M\n");
        }
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    for l in 0..2 {
                        for m in 0..2 {
                            for n in 0..2 {
                                alltrue = alltrue
                                    && arkode_butcher_order6n(
                                        b[i], A[j], c[k], A[l], c[m], c[n], s,
                                    );
                            }
                        }
                    }
                }
            }
        }
        if !alltrue && !outfile.is_null() {
            outfile.write_str("  method fails order 6 conditions N\n");
        }
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    for l in 0..2 {
                        for m in 0..2 {
                            for n in 0..2 {
                                alltrue = alltrue
                                    && arkode_butcher_order6o(
                                        b[i], A[j], c[k], A[l], A[m], c[n], s,
                                    );
                            }
                        }
                    }
                }
            }
        }
        if !alltrue && !outfile.is_null() {
            outfile.write_str("  method fails order 6 conditions O\n");
        }
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    for l in 0..2 {
                        for m in 0..2 {
                            for n in 0..2 {
                                alltrue = alltrue
                                    && arkode_butcher_order6p(
                                        b[i], A[j], A[k], c[l], c[m], c[n], s,
                                    );
                            }
                        }
                    }
                }
            }
        }
        if !alltrue && !outfile.is_null() {
            outfile.write_str("  method fails order 6 conditions P\n");
        }
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    for l in 0..2 {
                        for m in 0..2 {
                            for n in 0..2 {
                                alltrue = alltrue
                                    && arkode_butcher_order6q(
                                        b[i], A[j], A[k], c[l], A[m], c[n], s,
                                    );
                            }
                        }
                    }
                }
            }
        }
        if !alltrue && !outfile.is_null() {
            outfile.write_str("  method fails order 6 conditions Q\n");
        }
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    for l in 0..2 {
                        for m in 0..2 {
                            for n in 0..2 {
                                alltrue = alltrue
                                    && arkode_butcher_order6r(
                                        b[i], A[j], A[k], A[l], c[m], c[n], s,
                                    );
                            }
                        }
                    }
                }
            }
        }
        if !alltrue && !outfile.is_null() {
            outfile.write_str("  method fails order 6 conditions R\n");
        }
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    for l in 0..2 {
                        for m in 0..2 {
                            for n in 0..2 {
                                alltrue = alltrue
                                    && arkode_butcher_order6s(
                                        b[i], A[j], A[k], A[l], A[m], c[n], s,
                                    );
                            }
                        }
                    }
                }
            }
        }
        if !alltrue && !outfile.is_null() {
            outfile.write_str("  method fails order 6 conditions S\n");
        }
        if alltrue {
            *q = 6;
        }
    }

    /* check embedding order */
    if !d[0].is_empty() && !d[1].is_empty() {
        if !outfile.is_null() {
            outfile.write_str("\n");
        }

        /*    row sum conditions */
        if arkode_butcher_rowsum(A[0], c[0], s) && arkode_butcher_rowsum(A[1], c[1], s) {
            *p = 0;
        } else {
            *p = -1;
            if !outfile.is_null() {
                outfile.write_str("  embedding fails row sum conditions\n");
            }
        }
        /*    order 1 conditions */
        if *p == 0 {
            if arkode_butcher_order1(d[0], s) && arkode_butcher_order1(d[1], s) {
                *p = 1;
            } else if !outfile.is_null() {
                outfile.write_str("  embedding fails order 1 conditions\n");
            }
        }
        /*    order 2 conditions */
        if *p == 1 {
            alltrue = SUNTRUE;
            for i in 0..2 {
                for j in 0..2 {
                    alltrue = alltrue && arkode_butcher_order2(d[i], c[j], s);
                }
            }
            if alltrue {
                *p = 2;
            } else if !outfile.is_null() {
                outfile.write_str("  embedding fails order 2 conditions\n");
            }
        }
        /*    order 3 conditions */
        if *p == 2 {
            alltrue = SUNTRUE;
            for i in 0..2 {
                for j in 0..2 {
                    for k in 0..2 {
                        alltrue = alltrue && arkode_butcher_order3a(d[i], c[j], c[k], s);
                    }
                }
            }
            if !alltrue && !outfile.is_null() {
                outfile.write_str("  embedding fails order 3 conditions A\n");
            }
            for i in 0..2 {
                for j in 0..2 {
                    for k in 0..2 {
                        alltrue = alltrue && arkode_butcher_order3b(d[i], A[j], c[k], s);
                    }
                }
            }
            if !alltrue && !outfile.is_null() {
                outfile.write_str("  embedding fails order 3 conditions B\n");
            }
            if alltrue {
                *p = 3;
            }
        }
        /*    order 4 conditions */
        if *p == 3 {
            alltrue = SUNTRUE;
            for i in 0..2 {
                for j in 0..2 {
                    for k in 0..2 {
                        for l in 0..2 {
                            alltrue =
                                alltrue && arkode_butcher_order4a(d[i], c[j], c[k], c[l], s);
                        }
                    }
                }
            }
            if !alltrue && !outfile.is_null() {
                outfile.write_str("  embedding fails order 4 conditions A\n");
            }
            for i in 0..2 {
                for j in 0..2 {
                    for k in 0..2 {
                        for l in 0..2 {
                            alltrue =
                                alltrue && arkode_butcher_order4b(d[i], c[j], A[k], c[l], s);
                        }
                    }
                }
            }
            if !alltrue && !outfile.is_null() {
                outfile.write_str("  embedding fails order 4 conditions B\n");
            }
            for i in 0..2 {
                for j in 0..2 {
                    for k in 0..2 {
                        for l in 0..2 {
                            alltrue =
                                alltrue && arkode_butcher_order4c(d[i], A[j], c[k], c[l], s);
                        }
                    }
                }
            }
            if !alltrue && !outfile.is_null() {
                outfile.write_str("  embedding fails order 4 conditions C\n");
            }
            for i in 0..2 {
                for j in 0..2 {
                    for k in 0..2 {
                        for l in 0..2 {
                            alltrue =
                                alltrue && arkode_butcher_order4d(d[i], A[j], A[k], c[l], s);
                        }
                    }
                }
            }
            if !alltrue && !outfile.is_null() {
                outfile.write_str("  embedding fails order 4 conditions D\n");
            }
            if alltrue {
                *p = 4;
            }
        }
        /*    order 5 conditions */
        if *p == 4 {
            alltrue = SUNTRUE;
            for i in 0..2 {
                for j in 0..2 {
                    for k in 0..2 {
                        for l in 0..2 {
                            for m in 0..2 {
                                alltrue = alltrue
                                    && arkode_butcher_order5a(d[i], c[j], c[k], c[l], c[m], s);
                            }
                        }
                    }
                }
            }
            if !alltrue && !outfile.is_null() {
                outfile.write_str("  embedding fails order 5 conditions A\n");
            }
            for i in 0..2 {
                for j in 0..2 {
                    for k in 0..2 {
                        for l in 0..2 {
                            for m in 0..2 {
                                alltrue = alltrue
                                    && arkode_butcher_order5b(d[i], c[j], c[k], A[l], c[m], s);
                            }
                        }
                    }
                }
            }
            if !alltrue && !outfile.is_null() {
                outfile.write_str("  embedding fails order 5 conditions B\n");
            }
            for i in 0..2 {
                for j in 0..2 {
                    for k in 0..2 {
                        for l in 0..2 {
                            for m in 0..2 {
                                alltrue = alltrue
                                    && arkode_butcher_order5c(d[i], A[j], c[k], A[l], c[m], s);
                            }
                        }
                    }
                }
            }
            if !alltrue && !outfile.is_null() {
                outfile.write_str("  embedding fails order 5 conditions C\n");
            }
            for i in 0..2 {
                for j in 0..2 {
                    for k in 0..2 {
                        for l in 0..2 {
                            for m in 0..2 {
                                alltrue = alltrue
                                    && arkode_butcher_order5d(d[i], c[j], A[k], c[l], c[m], s);
                            }
                        }
                    }
                }
            }
            if !alltrue && !outfile.is_null() {
                outfile.write_str("  embedding fails order 5 conditions D\n");
            }
            for i in 0..2 {
                for j in 0..2 {
                    for k in 0..2 {
                        for l in 0..2 {
                            for m in 0..2 {
                                alltrue = alltrue
                                    && arkode_butcher_order5e(d[i], A[j], c[k], c[l], c[m], s);
                            }
                        }
                    }
                }
            }
            if !alltrue && !outfile.is_null() {
                outfile.write_str("  embedding fails order 5 conditions E\n");
            }
            for i in 0..2 {
                for j in 0..2 {
                    for k in 0..2 {
                        for l in 0..2 {
                            for m in 0..2 {
                                alltrue = alltrue
                                    && arkode_butcher_order5f(d[i], c[j], A[k], A[l], c[m], s);
                            }
                        }
                    }
                }
            }
            if !alltrue && !outfile.is_null() {
                outfile.write_str("  embedding fails order 5 conditions F\n");
            }
            for i in 0..2 {
                for j in 0..2 {
                    for k in 0..2 {
                        for l in 0..2 {
                            for m in 0..2 {
                                alltrue = alltrue
                                    && arkode_butcher_order5g(d[i], A[j], c[k], A[l], c[m], s);
                            }
                        }
                    }
                }
            }
            if !alltrue && !outfile.is_null() {
                outfile.write_str("  embedding fails order 5 conditions G\n");
            }
            for i in 0..2 {
                for j in 0..2 {
                    for k in 0..2 {
                        for l in 0..2 {
                            for m in 0..2 {
                                alltrue = alltrue
                                    && arkode_butcher_order5h(d[i], A[j], A[k], c[l], c[m], s);
                            }
                        }
                    }
                }
            }
            if !alltrue && !outfile.is_null() {
                outfile.write_str("  embedding fails order 5 conditions H\n");
            }
            for i in 0..2 {
                for j in 0..2 {
                    for k in 0..2 {
                        for l in 0..2 {
                            for m in 0..2 {
                                alltrue = alltrue
                                    && arkode_butcher_order5i(d[i], A[j], A[k], A[l], c[m], s);
                            }
                        }
                    }
                }
            }
            if !alltrue && !outfile.is_null() {
                outfile.write_str("  embedding fails order 5 conditions I\n");
            }
            if alltrue {
                *p = 5;
            }
        }
        /*    order 6 conditions */
        if *p == 5 {
            alltrue = SUNTRUE;
            for i in 0..2 {
                for j in 0..2 {
                    for k in 0..2 {
                        for l in 0..2 {
                            for m in 0..2 {
                                for n in 0..2 {
                                    alltrue = alltrue
                                        && arkode_butcher_order6a(
                                            d[i], c[j], c[k], c[l], c[m], c[n], s,
                                        );
                                }
                            }
                        }
                    }
                }
            }
            if !alltrue && !outfile.is_null() {
                outfile.write_str("  embedding fails order 6 conditions A\n");
            }
            for i in 0..2 {
                for j in 0..2 {
                    for k in 0..2 {
                        for l in 0..2 {
                            for m in 0..2 {
                                for n in 0..2 {
                                    alltrue = alltrue
                                        && arkode_butcher_order6b(
                                            d[i], c[j], c[k], c[l], A[m], c[n], s,
                                        );
                                }
                            }
                        }
                    }
                }
            }
            if !alltrue && !outfile.is_null() {
                outfile.write_str("  embedding fails order 6 conditions B\n");
            }
            for i in 0..2 {
                for j in 0..2 {
                    for k in 0..2 {
                        for l in 0..2 {
                            for m in 0..2 {
                                for n in 0..2 {
                                    alltrue = alltrue
                                        && arkode_butcher_order6c(
                                            d[i], c[j], A[k], c[l], A[m], c[n], s,
                                        );
                                }
                            }
                        }
                    }
                }
            }
            if !alltrue && !outfile.is_null() {
                outfile.write_str("  embedding fails order 6 conditions C\n");
            }
            for i in 0..2 {
                for j in 0..2 {
                    for k in 0..2 {
                        for l in 0..2 {
                            for m in 0..2 {
                                for n in 0..2 {
                                    alltrue = alltrue
                                        && arkode_butcher_order6d(
                                            d[i], c[j], c[k], A[l], c[m], c[n], s,
                                        );
                                }
                            }
                        }
                    }
                }
            }
            if !alltrue && !outfile.is_null() {
                outfile.write_str("  embedding fails order 6 conditions D\n");
            }
            for i in 0..2 {
                for j in 0..2 {
                    for k in 0..2 {
                        for l in 0..2 {
                            for m in 0..2 {
                                for n in 0..2 {
                                    alltrue = alltrue
                                        && arkode_butcher_order6e(
                                            d[i], c[j], c[k], A[l], A[m], c[n], s,
                                        );
                                }
                            }
                        }
                    }
                }
            }
            if !alltrue && !outfile.is_null() {
                outfile.write_str("  embedding fails order 6 conditions E\n");
            }
            for i in 0..2 {
                for j in 0..2 {
                    for k in 0..2 {
                        for l in 0..2 {
                            for m in 0..2 {
                                for n in 0..2 {
                                    alltrue = alltrue
                                        && arkode_butcher_order6f(
                                            d[i], A[j], A[k], c[l], A[m], c[n], s,
                                        );
                                }
                            }
                        }
                    }
                }
            }
            if !alltrue && !outfile.is_null() {
                outfile.write_str("  embedding fails order 6 conditions F\n");
            }
            for i in 0..2 {
                for j in 0..2 {
                    for k in 0..2 {
                        for l in 0..2 {
                            for m in 0..2 {
                                for n in 0..2 {
                                    alltrue = alltrue
                                        && arkode_butcher_order6g(
                                            d[i], c[j], A[k], c[l], c[m], c[n], s,
                                        );
                                }
                            }
                        }
                    }
                }
            }
            if !alltrue && !outfile.is_null() {
                outfile.write_str("  embedding fails order 6 conditions G\n");
            }
            for i in 0..2 {
                for j in 0..2 {
                    for k in 0..2 {
                        for l in 0..2 {
                            for m in 0..2 {
                                for n in 0..2 {
                                    alltrue = alltrue
                                        && arkode_butcher_order6h(
                                            d[i], c[j], A[k], c[l], A[m], c[n], s,
                                        );
                                }
                            }
                        }
                    }
                }
            }
            if !alltrue && !outfile.is_null() {
                outfile.write_str("  embedding fails order 6 conditions H\n");
            }
            for i in 0..2 {
                for j in 0..2 {
                    for k in 0..2 {
                        for l in 0..2 {
                            for m in 0..2 {
                                for n in 0..2 {
                                    alltrue = alltrue
                                        && arkode_butcher_order6i(
                                            d[i], c[j], A[k], A[l], c[m], c[n], s,
                                        );
                                }
                            }
                        }
                    }
                }
            }
            if !alltrue && !outfile.is_null() {
                outfile.write_str("  embedding fails order 6 conditions I\n");
            }
            for i in 0..2 {
                for j in 0..2 {
                    for k in 0..2 {
                        for l in 0..2 {
                            for m in 0..2 {
                                for n in 0..2 {
                                    alltrue = alltrue
                                        && arkode_butcher_order6j(
                                            d[i], c[j], A[k], A[l], A[m], c[n], s,
                                        );
                                }
                            }
                        }
                    }
                }
            }
            if !alltrue && !outfile.is_null() {
                outfile.write_str("  embedding fails order 6 conditions J\n");
            }
            for i in 0..2 {
                for j in 0..2 {
                    for k in 0..2 {
                        for l in 0..2 {
                            for m in 0..2 {
                                for n in 0..2 {
                                    alltrue = alltrue
                                        && arkode_butcher_order6k(
                                            d[i], A[j], c[k], c[l], c[m], c[n], s,
                                        );
                                }
                            }
                        }
                    }
                }
            }
            if !alltrue && !outfile.is_null() {
                outfile.write_str("  embedding fails order 6 conditions K\n");
            }
            for i in 0..2 {
                for j in 0..2 {
                    for k in 0..2 {
                        for l in 0..2 {
                            for m in 0..2 {
                                for n in 0..2 {
                                    alltrue = alltrue
                                        && arkode_butcher_order6l(
                                            d[i], A[j], c[k], c[l], A[m], c[n], s,
                                        );
                                }
                            }
                        }
                    }
                }
            }
            if !alltrue && !outfile.is_null() {
                outfile.write_str("  embedding fails order 6 conditions L\n");
            }
            for i in 0..2 {
                for j in 0..2 {
                    for k in 0..2 {
                        for l in 0..2 {
                            for m in 0..2 {
                                for n in 0..2 {
                                    alltrue = alltrue
                                        && arkode_butcher_order6m(
                                            d[i], A[j], A[k], c[l], A[m], c[n], s,
                                        );
                                }
                            }
                        }
                    }
                }
            }
            if !alltrue && !outfile.is_null() {
                outfile.write_str("  embedding fails order 6 conditions M\n");
            }
            for i in 0..2 {
                for j in 0..2 {
                    for k in 0..2 {
                        for l in 0..2 {
                            for m in 0..2 {
                                for n in 0..2 {
                                    alltrue = alltrue
                                        && arkode_butcher_order6n(
                                            d[i], A[j], c[k], A[l], c[m], c[n], s,
                                        );
                                }
                            }
                        }
                    }
                }
            }
            if !alltrue && !outfile.is_null() {
                outfile.write_str("  embedding fails order 6 conditions N\n");
            }
            for i in 0..2 {
                for j in 0..2 {
                    for k in 0..2 {
                        for l in 0..2 {
                            for m in 0..2 {
                                for n in 0..2 {
                                    alltrue = alltrue
                                        && arkode_butcher_order6o(
                                            d[i], A[j], c[k], A[l], A[m], c[n], s,
                                        );
                                }
                            }
                        }
                    }
                }
            }
            if !alltrue && !outfile.is_null() {
                outfile.write_str("  embedding fails order 6 conditions O\n");
            }
            for i in 0..2 {
                for j in 0..2 {
                    for k in 0..2 {
                        for l in 0..2 {
                            for m in 0..2 {
                                for n in 0..2 {
                                    alltrue = alltrue
                                        && arkode_butcher_order6p(
                                            d[i], A[j], A[k], c[l], c[m], c[n], s,
                                        );
                                }
                            }
                        }
                    }
                }
            }
            if !alltrue && !outfile.is_null() {
                outfile.write_str("  embedding fails order 6 conditions P\n");
            }
            for i in 0..2 {
                for j in 0..2 {
                    for k in 0..2 {
                        for l in 0..2 {
                            for m in 0..2 {
                                for n in 0..2 {
                                    alltrue = alltrue
                                        && arkode_butcher_order6q(
                                            d[i], A[j], A[k], c[l], A[m], c[n], s,
                                        );
                                }
                            }
                        }
                    }
                }
            }
            if !alltrue && !outfile.is_null() {
                outfile.write_str("  embedding fails order 6 conditions Q\n");
            }
            for i in 0..2 {
                for j in 0..2 {
                    for k in 0..2 {
                        for l in 0..2 {
                            for m in 0..2 {
                                for n in 0..2 {
                                    alltrue = alltrue
                                        && arkode_butcher_order6r(
                                            d[i], A[j], A[k], A[l], c[m], c[n], s,
                                        );
                                }
                            }
                        }
                    }
                }
            }
            if !alltrue && !outfile.is_null() {
                outfile.write_str("  embedding fails order 6 conditions R\n");
            }
            for i in 0..2 {
                for j in 0..2 {
                    for k in 0..2 {
                        for l in 0..2 {
                            for m in 0..2 {
                                for n in 0..2 {
                                    alltrue = alltrue
                                        && arkode_butcher_order6s(
                                            d[i], A[j], A[k], A[l], A[m], c[n], s,
                                        );
                                }
                            }
                        }
                    }
                }
            }
            if !alltrue && !outfile.is_null() {
                outfile.write_str("  embedding fails order 6 conditions S\n");
            }
            if alltrue {
                *p = 6;
            }
        }
    }

    /* compare results against stored values and return */

    /*    check warning modes */
    if *q > B1m.q {
        return 1;
    }
    if *q > B2m.q {
        return 1;
    }
    if !d[0].is_empty() && !d[1].is_empty() {
        if *p > B1m.p {
            return 1;
        }
        if *p > B2m.p {
            return 1;
        }
    }
    if (*q < B1m.q) && (*q == 6) {
        return 1;
    }
    if (*q < B2m.q) && (*q == 6) {
        return 1;
    }
    if !d[0].is_empty() && !d[1].is_empty() {
        if (*p < B1m.p) && (*p == 6) {
            return 1;
        }
        if (*p < B2m.p) && (*p == 6) {
            return 1;
        }
    }

    /*    return success */
    0
}

/*---------------------------------------------------------------
  Private utility routines for checking method order
  ---------------------------------------------------------------*/

/// C `calloc(s, sizeof(sunrealtype))`. Both public entry points reject
/// `stages < 1`, so `s >= 1` at every call site; the clamp only keeps a
/// hypothetical negative `s` from becoming a huge allocation.
#[inline]
fn butcher_tmp(s: i32) -> Vec<sunrealtype> {
    vec![0.0; if s > 0 { s as usize } else { 0 }]
}

/*---------------------------------------------------------------
  Utility routine to compute small dense matrix-vector product
       b = A*x
  Here A is (s x s), x and b are (s x 1).  Returns 0 on success,
  nonzero on failure.
  ---------------------------------------------------------------*/
fn arkode_butcher_mv(
    A: &[Vec<sunrealtype>],
    x: &[sunrealtype],
    s: i32,
    b: &mut [sunrealtype],
) -> i32 {
    /* the C NULL-pointer guards cannot be expressed for slices */
    if s < 1 {
        return 1;
    }
    for i in 0..s {
        b[i as usize] = 0.0;
    }
    for i in 0..s {
        for j in 0..s {
            b[i as usize] += A[i as usize][j as usize] * x[j as usize];
        }
    }
    0
}

/*---------------------------------------------------------------
  Utility routine to compute small vector .* vector product
       z = x.*y   [Matlab notation]
  Here all vectors are (s x 1).   Returns 0 on success,
  nonzero on failure.
  ---------------------------------------------------------------*/
fn arkode_butcher_vv(
    x: &[sunrealtype],
    y: &[sunrealtype],
    s: i32,
    z: &mut [sunrealtype],
) -> i32 {
    if s < 1 {
        return 1;
    }
    for i in 0..s {
        z[i as usize] = x[i as usize] * y[i as usize];
    }
    0
}

/*---------------------------------------------------------------
  Utility routine to compute small vector .^ int
       z = x.^l   [Matlab notation]
  Here all vectors are (s x 1).   Returns 0 on success,
  nonzero on failure.
  ---------------------------------------------------------------*/
fn arkode_butcher_vp(x: &[sunrealtype], l: i32, s: i32, z: &mut [sunrealtype]) -> i32 {
    if s < 1 {
        return 1;
    }
    for i in 0..s {
        z[i as usize] = SUNRpowerI(x[i as usize], l);
    }
    0
}

/*---------------------------------------------------------------
  Utility routine to compute small vector dot product:
       d = dot(x,y)
  Here x and y are (s x 1), and d is scalar.   Returns 0 on success,
  nonzero on failure.
  ---------------------------------------------------------------*/
fn arkode_butcher_dot(
    x: &[sunrealtype],
    y: &[sunrealtype],
    s: i32,
    d: &mut sunrealtype,
) -> i32 {
    if s < 1 {
        return 1;
    }
    *d = 0.0;
    for i in 0..s {
        *d += x[i as usize] * y[i as usize];
    }
    0
}

/*---------------------------------------------------------------
  Utility routines to check specific order conditions.  Each
  returns SUNTRUE on success, SUNFALSE on failure.
     Order 0:  arkode_butcher_rowsum
     Order 1:  arkode_butcher_order1
     Order 2:  arkode_butcher_order2
     Order 3:  arkode_butcher_order3a and arkode_butcher_order3b
     Order 4:  arkode_butcher_order4a through arkode_butcher_order4d
     Order 5:  arkode_butcher_order5a through arkode_butcher_order5i
     Order 6:  arkode_butcher_order6a through arkode_butcher_order6s
  ---------------------------------------------------------------*/

/* c(i) = sum(A(i,:)) */
fn arkode_butcher_rowsum(A: &[Vec<sunrealtype>], c: &[sunrealtype], s: i32) -> sunbooleantype {
    for i in 0..s {
        let mut rsum: sunrealtype = 0.0;
        for j in 0..s {
            rsum += A[i as usize][j as usize];
        }
        if SUNRabs(rsum - c[i as usize]) > TOL() {
            return SUNFALSE;
        }
    }
    SUNTRUE
}

/* b'*e = 1 */
fn arkode_butcher_order1(b: &[sunrealtype], s: i32) -> sunbooleantype {
    let mut err: sunrealtype = 1.0;
    for i in 0..s {
        err -= b[i as usize];
    }
    if SUNRabs(err) > TOL() {
        SUNFALSE
    } else {
        SUNTRUE
    }
}

/* b'*c = 1/2 */
fn arkode_butcher_order2(b: &[sunrealtype], c: &[sunrealtype], s: i32) -> sunbooleantype {
    let mut bc: sunrealtype = 0.0;
    if arkode_butcher_dot(b, c, s, &mut bc) != 0 {
        return SUNFALSE;
    }
    if SUNRabs(bc - 0.5) > TOL() {
        SUNFALSE
    } else {
        SUNTRUE
    }
}

/* b'*(c1.*c2) = 1/3 */
fn arkode_butcher_order3a(
    b: &[sunrealtype],
    c1: &[sunrealtype],
    c2: &[sunrealtype],
    s: i32,
) -> sunbooleantype {
    let mut bcc: sunrealtype = 0.0;
    let mut tmp = butcher_tmp(s);
    if arkode_butcher_vv(c1, c2, s, &mut tmp) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_dot(b, &tmp, s, &mut bcc) != 0 {
        return SUNFALSE;
    }
    if SUNRabs(bcc - 1.0 / 3.0) > TOL() {
        SUNFALSE
    } else {
        SUNTRUE
    }
}

/* b'*(A*c) = 1/6 */
fn arkode_butcher_order3b(
    b: &[sunrealtype],
    A: &[Vec<sunrealtype>],
    c: &[sunrealtype],
    s: i32,
) -> sunbooleantype {
    let mut bAc: sunrealtype = 0.0;
    let mut tmp = butcher_tmp(s);
    if arkode_butcher_mv(A, c, s, &mut tmp) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_dot(b, &tmp, s, &mut bAc) != 0 {
        return SUNFALSE;
    }
    if SUNRabs(bAc - 1.0 / 6.0) > TOL() {
        SUNFALSE
    } else {
        SUNTRUE
    }
}

/* b'*(c1.*c2.*c3) = 1/4 */
fn arkode_butcher_order4a(
    b: &[sunrealtype],
    c1: &[sunrealtype],
    c2: &[sunrealtype],
    c3: &[sunrealtype],
    s: i32,
) -> sunbooleantype {
    let mut bccc: sunrealtype = 0.0;
    let mut tmp1 = butcher_tmp(s);
    let mut tmp2 = butcher_tmp(s);
    if arkode_butcher_vv(c1, c2, s, &mut tmp1) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_vv(c3, &tmp1, s, &mut tmp2) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_dot(b, &tmp2, s, &mut bccc) != 0 {
        return SUNFALSE;
    }
    if SUNRabs(bccc - 0.25) > TOL() {
        SUNFALSE
    } else {
        SUNTRUE
    }
}

/* (b.*c1)'*(A*c2) = 1/8 */
fn arkode_butcher_order4b(
    b: &[sunrealtype],
    c1: &[sunrealtype],
    A: &[Vec<sunrealtype>],
    c2: &[sunrealtype],
    s: i32,
) -> sunbooleantype {
    let mut bcAc: sunrealtype = 0.0;
    let mut tmp1 = butcher_tmp(s);
    let mut tmp2 = butcher_tmp(s);
    if arkode_butcher_vv(b, c1, s, &mut tmp1) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_mv(A, c2, s, &mut tmp2) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_dot(&tmp1, &tmp2, s, &mut bcAc) != 0 {
        return SUNFALSE;
    }
    if SUNRabs(bcAc - 0.125) > TOL() {
        SUNFALSE
    } else {
        SUNTRUE
    }
}

/* b'*A*(c1.*c2) = 1/12 */
fn arkode_butcher_order4c(
    b: &[sunrealtype],
    A: &[Vec<sunrealtype>],
    c1: &[sunrealtype],
    c2: &[sunrealtype],
    s: i32,
) -> sunbooleantype {
    let mut bAcc: sunrealtype = 0.0;
    let mut tmp1 = butcher_tmp(s);
    let mut tmp2 = butcher_tmp(s);
    if arkode_butcher_vv(c1, c2, s, &mut tmp1) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_mv(A, &tmp1, s, &mut tmp2) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_dot(b, &tmp2, s, &mut bAcc) != 0 {
        return SUNFALSE;
    }
    if SUNRabs(bAcc - 1.0 / 12.0) > TOL() {
        SUNFALSE
    } else {
        SUNTRUE
    }
}

/* b'*A1*A2*c = 1/24 */
fn arkode_butcher_order4d(
    b: &[sunrealtype],
    A1: &[Vec<sunrealtype>],
    A2: &[Vec<sunrealtype>],
    c: &[sunrealtype],
    s: i32,
) -> sunbooleantype {
    let mut bAAc: sunrealtype = 0.0;
    let mut tmp1 = butcher_tmp(s);
    let mut tmp2 = butcher_tmp(s);
    if arkode_butcher_mv(A2, c, s, &mut tmp1) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_mv(A1, &tmp1, s, &mut tmp2) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_dot(b, &tmp2, s, &mut bAAc) != 0 {
        return SUNFALSE;
    }
    if SUNRabs(bAAc - 1.0 / 24.0) > TOL() {
        SUNFALSE
    } else {
        SUNTRUE
    }
}

/* b'*(c1.*c2.*c3.*c4) = 1/5 */
fn arkode_butcher_order5a(
    b: &[sunrealtype],
    c1: &[sunrealtype],
    c2: &[sunrealtype],
    c3: &[sunrealtype],
    c4: &[sunrealtype],
    s: i32,
) -> sunbooleantype {
    let mut bcccc: sunrealtype = 0.0;
    let mut tmp1 = butcher_tmp(s);
    let mut tmp2 = butcher_tmp(s);
    if arkode_butcher_vv(c1, c2, s, &mut tmp1) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_vv(c3, &tmp1, s, &mut tmp2) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_vv(c4, &tmp2, s, &mut tmp1) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_dot(b, &tmp1, s, &mut bcccc) != 0 {
        return SUNFALSE;
    }
    if SUNRabs(bcccc - 0.2) > TOL() {
        SUNFALSE
    } else {
        SUNTRUE
    }
}

/* (b.*c1.*c2)'*(A*c3) = 1/10 */
fn arkode_butcher_order5b(
    b: &[sunrealtype],
    c1: &[sunrealtype],
    c2: &[sunrealtype],
    A: &[Vec<sunrealtype>],
    c3: &[sunrealtype],
    s: i32,
) -> sunbooleantype {
    let mut bccAc: sunrealtype = 0.0;
    let mut tmp1 = butcher_tmp(s);
    let mut tmp2 = butcher_tmp(s);
    if arkode_butcher_vv(c1, c2, s, &mut tmp1) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_vv(b, &tmp1, s, &mut tmp2) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_mv(A, c3, s, &mut tmp1) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_dot(&tmp1, &tmp2, s, &mut bccAc) != 0 {
        return SUNFALSE;
    }
    if SUNRabs(bccAc - 0.1) > TOL() {
        SUNFALSE
    } else {
        SUNTRUE
    }
}

/* b'*((A1*c1).*(A2*c2)) = 1/20 */
fn arkode_butcher_order5c(
    b: &[sunrealtype],
    A1: &[Vec<sunrealtype>],
    c1: &[sunrealtype],
    A2: &[Vec<sunrealtype>],
    c2: &[sunrealtype],
    s: i32,
) -> sunbooleantype {
    let mut bAcAc: sunrealtype = 0.0;
    let mut tmp1 = butcher_tmp(s);
    let mut tmp2 = butcher_tmp(s);
    let mut tmp3 = butcher_tmp(s);
    if arkode_butcher_mv(A1, c1, s, &mut tmp1) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_mv(A2, c2, s, &mut tmp2) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_vv(&tmp1, &tmp2, s, &mut tmp3) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_dot(b, &tmp3, s, &mut bAcAc) != 0 {
        return SUNFALSE;
    }
    if SUNRabs(bAcAc - 0.05) > TOL() {
        SUNFALSE
    } else {
        SUNTRUE
    }
}

/* (b.*c1)'*A*(c2.*c3) = 1/15 */
fn arkode_butcher_order5d(
    b: &[sunrealtype],
    c1: &[sunrealtype],
    A: &[Vec<sunrealtype>],
    c2: &[sunrealtype],
    c3: &[sunrealtype],
    s: i32,
) -> sunbooleantype {
    let mut bcAcc: sunrealtype = 0.0;
    let mut tmp1 = butcher_tmp(s);
    let mut tmp2 = butcher_tmp(s);
    if arkode_butcher_vv(c2, c3, s, &mut tmp1) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_mv(A, &tmp1, s, &mut tmp2) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_vv(b, c1, s, &mut tmp1) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_dot(&tmp1, &tmp2, s, &mut bcAcc) != 0 {
        return SUNFALSE;
    }
    if SUNRabs(bcAcc - 1.0 / 15.0) > TOL() {
        SUNFALSE
    } else {
        SUNTRUE
    }
}

/* b'*A*(c1.*c2.*c3) = 1/20 */
fn arkode_butcher_order5e(
    b: &[sunrealtype],
    A: &[Vec<sunrealtype>],
    c1: &[sunrealtype],
    c2: &[sunrealtype],
    c3: &[sunrealtype],
    s: i32,
) -> sunbooleantype {
    let mut bAccc: sunrealtype = 0.0;
    let mut tmp1 = butcher_tmp(s);
    let mut tmp2 = butcher_tmp(s);
    if arkode_butcher_vv(c1, c2, s, &mut tmp1) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_vv(c3, &tmp1, s, &mut tmp2) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_mv(A, &tmp2, s, &mut tmp1) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_dot(b, &tmp1, s, &mut bAccc) != 0 {
        return SUNFALSE;
    }
    if SUNRabs(bAccc - 0.05) > TOL() {
        SUNFALSE
    } else {
        SUNTRUE
    }
}

/* (b.*c1)'*A1*A2*c2 = 1/30 */
fn arkode_butcher_order5f(
    b: &[sunrealtype],
    c1: &[sunrealtype],
    A1: &[Vec<sunrealtype>],
    A2: &[Vec<sunrealtype>],
    c2: &[sunrealtype],
    s: i32,
) -> sunbooleantype {
    let mut bcAAc: sunrealtype = 0.0;
    let mut tmp1 = butcher_tmp(s);
    let mut tmp2 = butcher_tmp(s);
    if arkode_butcher_mv(A2, c2, s, &mut tmp1) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_mv(A1, &tmp1, s, &mut tmp2) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_vv(b, c1, s, &mut tmp1) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_dot(&tmp1, &tmp2, s, &mut bcAAc) != 0 {
        return SUNFALSE;
    }
    if SUNRabs(bcAAc - 1.0 / 30.0) > TOL() {
        SUNFALSE
    } else {
        SUNTRUE
    }
}

/* b'*A1*(c1.*(A2*c2)) = 1/40 */
fn arkode_butcher_order5g(
    b: &[sunrealtype],
    A1: &[Vec<sunrealtype>],
    c1: &[sunrealtype],
    A2: &[Vec<sunrealtype>],
    c2: &[sunrealtype],
    s: i32,
) -> sunbooleantype {
    let mut bAcAc: sunrealtype = 0.0;
    let mut tmp1 = butcher_tmp(s);
    let mut tmp2 = butcher_tmp(s);
    if arkode_butcher_mv(A2, c2, s, &mut tmp1) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_vv(c1, &tmp1, s, &mut tmp2) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_mv(A1, &tmp2, s, &mut tmp1) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_dot(b, &tmp1, s, &mut bAcAc) != 0 {
        return SUNFALSE;
    }
    if SUNRabs(bAcAc - 1.0 / 40.0) > TOL() {
        SUNFALSE
    } else {
        SUNTRUE
    }
}

/* b'*A1*A2*(c1.*c2) = 1/60 */
fn arkode_butcher_order5h(
    b: &[sunrealtype],
    A1: &[Vec<sunrealtype>],
    A2: &[Vec<sunrealtype>],
    c1: &[sunrealtype],
    c2: &[sunrealtype],
    s: i32,
) -> sunbooleantype {
    let mut bAAcc: sunrealtype = 0.0;
    let mut tmp1 = butcher_tmp(s);
    let mut tmp2 = butcher_tmp(s);
    if arkode_butcher_vv(c1, c2, s, &mut tmp1) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_mv(A2, &tmp1, s, &mut tmp2) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_mv(A1, &tmp2, s, &mut tmp1) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_dot(b, &tmp1, s, &mut bAAcc) != 0 {
        return SUNFALSE;
    }
    if SUNRabs(bAAcc - 1.0 / 60.0) > TOL() {
        SUNFALSE
    } else {
        SUNTRUE
    }
}

/* b'*A1*A2*A3*c = 1/120 */
fn arkode_butcher_order5i(
    b: &[sunrealtype],
    A1: &[Vec<sunrealtype>],
    A2: &[Vec<sunrealtype>],
    A3: &[Vec<sunrealtype>],
    c: &[sunrealtype],
    s: i32,
) -> sunbooleantype {
    let mut bAAAc: sunrealtype = 0.0;
    let mut tmp1 = butcher_tmp(s);
    let mut tmp2 = butcher_tmp(s);
    if arkode_butcher_mv(A3, c, s, &mut tmp1) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_mv(A2, &tmp1, s, &mut tmp2) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_mv(A1, &tmp2, s, &mut tmp1) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_dot(b, &tmp1, s, &mut bAAAc) != 0 {
        return SUNFALSE;
    }
    if SUNRabs(bAAAc - 1.0 / 120.0) > TOL() {
        SUNFALSE
    } else {
        SUNTRUE
    }
}

/* b'*(c1.*c2.*c3.*c4.*c5) = 1/6 */
fn arkode_butcher_order6a(
    b: &[sunrealtype],
    c1: &[sunrealtype],
    c2: &[sunrealtype],
    c3: &[sunrealtype],
    c4: &[sunrealtype],
    c5: &[sunrealtype],
    s: i32,
) -> sunbooleantype {
    let mut bccccc: sunrealtype = 0.0;
    let mut tmp1 = butcher_tmp(s);
    let mut tmp2 = butcher_tmp(s);
    if arkode_butcher_vv(c1, c2, s, &mut tmp1) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_vv(c3, &tmp1, s, &mut tmp2) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_vv(c4, &tmp2, s, &mut tmp1) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_vv(c5, &tmp1, s, &mut tmp2) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_dot(b, &tmp2, s, &mut bccccc) != 0 {
        return SUNFALSE;
    }
    if SUNRabs(bccccc - 1.0 / 6.0) > TOL() {
        SUNFALSE
    } else {
        SUNTRUE
    }
}

/* (b.*c1.*c2.*c3)'*(A*c4) = 1/12 */
fn arkode_butcher_order6b(
    b: &[sunrealtype],
    c1: &[sunrealtype],
    c2: &[sunrealtype],
    c3: &[sunrealtype],
    A: &[Vec<sunrealtype>],
    c4: &[sunrealtype],
    s: i32,
) -> sunbooleantype {
    let mut bcccAc: sunrealtype = 0.0;
    let mut tmp1 = butcher_tmp(s);
    let mut tmp2 = butcher_tmp(s);
    if arkode_butcher_vv(b, c1, s, &mut tmp1) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_vv(c2, &tmp1, s, &mut tmp2) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_vv(c3, &tmp2, s, &mut tmp1) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_mv(A, c4, s, &mut tmp2) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_dot(&tmp1, &tmp2, s, &mut bcccAc) != 0 {
        return SUNFALSE;
    }
    if SUNRabs(bcccAc - 1.0 / 12.0) > TOL() {
        SUNFALSE
    } else {
        SUNTRUE
    }
}

/* b'*(c1.*(A1*c2).*(A2*c3)) = 1/24 */
fn arkode_butcher_order6c(
    b: &[sunrealtype],
    c1: &[sunrealtype],
    A1: &[Vec<sunrealtype>],
    c2: &[sunrealtype],
    A2: &[Vec<sunrealtype>],
    c3: &[sunrealtype],
    s: i32,
) -> sunbooleantype {
    let mut bcAc2: sunrealtype = 0.0;
    let mut tmp1 = butcher_tmp(s);
    let mut tmp2 = butcher_tmp(s);
    let mut tmp3 = butcher_tmp(s);
    if arkode_butcher_mv(A2, c3, s, &mut tmp1) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_mv(A1, c2, s, &mut tmp2) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_vv(&tmp1, &tmp2, s, &mut tmp3) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_vv(c1, &tmp3, s, &mut tmp1) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_dot(b, &tmp1, s, &mut bcAc2) != 0 {
        return SUNFALSE;
    }
    if SUNRabs(bcAc2 - 1.0 / 24.0) > TOL() {
        SUNFALSE
    } else {
        SUNTRUE
    }
}

/* (b.*c1.*c2)'*A*(c3.*c4) = 1/18 */
fn arkode_butcher_order6d(
    b: &[sunrealtype],
    c1: &[sunrealtype],
    c2: &[sunrealtype],
    A: &[Vec<sunrealtype>],
    c3: &[sunrealtype],
    c4: &[sunrealtype],
    s: i32,
) -> sunbooleantype {
    let mut bccAcc: sunrealtype = 0.0;
    let mut tmp1 = butcher_tmp(s);
    let mut tmp2 = butcher_tmp(s);
    let mut tmp3 = butcher_tmp(s);
    if arkode_butcher_vv(c3, c4, s, &mut tmp1) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_mv(A, &tmp1, s, &mut tmp2) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_vv(c1, c2, s, &mut tmp1) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_vv(b, &tmp1, s, &mut tmp3) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_dot(&tmp2, &tmp3, s, &mut bccAcc) != 0 {
        return SUNFALSE;
    }
    if SUNRabs(bccAcc - 1.0 / 18.0) > TOL() {
        SUNFALSE
    } else {
        SUNTRUE
    }
}

/* (b.*(c1.*c2))'*A1*A2*c3 = 1/36 */
fn arkode_butcher_order6e(
    b: &[sunrealtype],
    c1: &[sunrealtype],
    c2: &[sunrealtype],
    A1: &[Vec<sunrealtype>],
    A2: &[Vec<sunrealtype>],
    c3: &[sunrealtype],
    s: i32,
) -> sunbooleantype {
    let mut bccAAc: sunrealtype = 0.0;
    let mut tmp1 = butcher_tmp(s);
    let mut tmp2 = butcher_tmp(s);
    let mut tmp3 = butcher_tmp(s);
    if arkode_butcher_vv(c1, c2, s, &mut tmp1) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_vv(b, &tmp1, s, &mut tmp2) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_mv(A2, c3, s, &mut tmp1) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_mv(A1, &tmp1, s, &mut tmp3) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_dot(&tmp2, &tmp3, s, &mut bccAAc) != 0 {
        return SUNFALSE;
    }
    if SUNRabs(bccAAc - 1.0 / 36.0) > TOL() {
        SUNFALSE
    } else {
        SUNTRUE
    }
}

/* b'*((A1*A2*c1).*(A3*c2)) = 1/72 */
fn arkode_butcher_order6f(
    b: &[sunrealtype],
    A1: &[Vec<sunrealtype>],
    A2: &[Vec<sunrealtype>],
    c1: &[sunrealtype],
    A3: &[Vec<sunrealtype>],
    c2: &[sunrealtype],
    s: i32,
) -> sunbooleantype {
    let mut bAAcAc: sunrealtype = 0.0;
    let mut tmp1 = butcher_tmp(s);
    let mut tmp2 = butcher_tmp(s);
    let mut tmp3 = butcher_tmp(s);
    if arkode_butcher_mv(A2, c1, s, &mut tmp1) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_mv(A1, &tmp1, s, &mut tmp2) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_mv(A3, c2, s, &mut tmp1) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_vv(&tmp1, &tmp2, s, &mut tmp3) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_dot(b, &tmp3, s, &mut bAAcAc) != 0 {
        return SUNFALSE;
    }
    if SUNRabs(bAAcAc - 1.0 / 72.0) > TOL() {
        SUNFALSE
    } else {
        SUNTRUE
    }
}

/* b'*(c1.*(A*(c2.*c3.*c4))) = 1/24 */
fn arkode_butcher_order6g(
    b: &[sunrealtype],
    c1: &[sunrealtype],
    A: &[Vec<sunrealtype>],
    c2: &[sunrealtype],
    c3: &[sunrealtype],
    c4: &[sunrealtype],
    s: i32,
) -> sunbooleantype {
    let mut bcAccc: sunrealtype = 0.0;
    let mut tmp1 = butcher_tmp(s);
    let mut tmp2 = butcher_tmp(s);
    if arkode_butcher_vv(c2, c3, s, &mut tmp1) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_vv(c4, &tmp1, s, &mut tmp2) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_mv(A, &tmp2, s, &mut tmp1) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_vv(c1, &tmp1, s, &mut tmp2) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_dot(b, &tmp2, s, &mut bcAccc) != 0 {
        return SUNFALSE;
    }
    if SUNRabs(bcAccc - 1.0 / 24.0) > TOL() {
        SUNFALSE
    } else {
        SUNTRUE
    }
}

/* b'*(c1.*(A1*(c2.*(A2*c3)))) = 1/48 */
fn arkode_butcher_order6h(
    b: &[sunrealtype],
    c1: &[sunrealtype],
    A1: &[Vec<sunrealtype>],
    c2: &[sunrealtype],
    A2: &[Vec<sunrealtype>],
    c3: &[sunrealtype],
    s: i32,
) -> sunbooleantype {
    let mut bcAcAc: sunrealtype = 0.0;
    let mut tmp1 = butcher_tmp(s);
    let mut tmp2 = butcher_tmp(s);
    if arkode_butcher_mv(A2, c3, s, &mut tmp1) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_vv(c2, &tmp1, s, &mut tmp2) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_mv(A1, &tmp2, s, &mut tmp1) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_vv(c1, &tmp1, s, &mut tmp2) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_dot(b, &tmp2, s, &mut bcAcAc) != 0 {
        return SUNFALSE;
    }
    if SUNRabs(bcAcAc - 1.0 / 48.0) > TOL() {
        SUNFALSE
    } else {
        SUNTRUE
    }
}

/* b'*(c1.*(A1*A2*(c2.*c3))) = 1/72 */
fn arkode_butcher_order6i(
    b: &[sunrealtype],
    c1: &[sunrealtype],
    A1: &[Vec<sunrealtype>],
    A2: &[Vec<sunrealtype>],
    c2: &[sunrealtype],
    c3: &[sunrealtype],
    s: i32,
) -> sunbooleantype {
    let mut bcAAcc: sunrealtype = 0.0;
    let mut tmp1 = butcher_tmp(s);
    let mut tmp2 = butcher_tmp(s);
    if arkode_butcher_vv(c2, c3, s, &mut tmp1) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_mv(A2, &tmp1, s, &mut tmp2) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_mv(A1, &tmp2, s, &mut tmp1) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_vv(c1, &tmp1, s, &mut tmp2) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_dot(b, &tmp2, s, &mut bcAAcc) != 0 {
        return SUNFALSE;
    }
    if SUNRabs(bcAAcc - 1.0 / 72.0) > TOL() {
        SUNFALSE
    } else {
        SUNTRUE
    }
}

/* b'*(c1.*(A1*A2*A3*c2)) = 1/144 */
fn arkode_butcher_order6j(
    b: &[sunrealtype],
    c1: &[sunrealtype],
    A1: &[Vec<sunrealtype>],
    A2: &[Vec<sunrealtype>],
    A3: &[Vec<sunrealtype>],
    c2: &[sunrealtype],
    s: i32,
) -> sunbooleantype {
    let mut bcAAAc: sunrealtype = 0.0;
    let mut tmp1 = butcher_tmp(s);
    let mut tmp2 = butcher_tmp(s);
    if arkode_butcher_mv(A3, c2, s, &mut tmp1) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_mv(A2, &tmp1, s, &mut tmp2) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_mv(A1, &tmp2, s, &mut tmp1) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_vv(c1, &tmp1, s, &mut tmp2) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_dot(b, &tmp2, s, &mut bcAAAc) != 0 {
        return SUNFALSE;
    }
    if SUNRabs(bcAAAc - 1.0 / 144.0) > TOL() {
        SUNFALSE
    } else {
        SUNTRUE
    }
}

/* b'*A*(c1.*c2.*c3.*c4) = 1/30 */
fn arkode_butcher_order6k(
    b: &[sunrealtype],
    A: &[Vec<sunrealtype>],
    c1: &[sunrealtype],
    c2: &[sunrealtype],
    c3: &[sunrealtype],
    c4: &[sunrealtype],
    s: i32,
) -> sunbooleantype {
    let mut bAcccc: sunrealtype = 0.0;
    let mut tmp1 = butcher_tmp(s);
    let mut tmp2 = butcher_tmp(s);
    if arkode_butcher_vv(c1, c2, s, &mut tmp1) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_vv(c3, &tmp1, s, &mut tmp2) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_vv(c4, &tmp2, s, &mut tmp1) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_mv(A, &tmp1, s, &mut tmp2) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_dot(b, &tmp2, s, &mut bAcccc) != 0 {
        return SUNFALSE;
    }
    if SUNRabs(bAcccc - 1.0 / 30.0) > TOL() {
        SUNFALSE
    } else {
        SUNTRUE
    }
}

/* b'*A1*(c1.*c2.*(A2*c3)) = 1/60 */
fn arkode_butcher_order6l(
    b: &[sunrealtype],
    A1: &[Vec<sunrealtype>],
    c1: &[sunrealtype],
    c2: &[sunrealtype],
    A2: &[Vec<sunrealtype>],
    c3: &[sunrealtype],
    s: i32,
) -> sunbooleantype {
    let mut bAccAc: sunrealtype = 0.0;
    let mut tmp1 = butcher_tmp(s);
    let mut tmp2 = butcher_tmp(s);
    if arkode_butcher_mv(A2, c3, s, &mut tmp1) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_vv(c2, &tmp1, s, &mut tmp2) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_vv(c1, &tmp2, s, &mut tmp1) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_mv(A1, &tmp1, s, &mut tmp2) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_dot(b, &tmp2, s, &mut bAccAc) != 0 {
        return SUNFALSE;
    }
    if SUNRabs(bAccAc - 1.0 / 60.0) > TOL() {
        SUNFALSE
    } else {
        SUNTRUE
    }
}

/* b'*A1*((A2*c1).*(A3*c2)) = 1/120 */
fn arkode_butcher_order6m(
    b: &[sunrealtype],
    A1: &[Vec<sunrealtype>],
    A2: &[Vec<sunrealtype>],
    c1: &[sunrealtype],
    A3: &[Vec<sunrealtype>],
    c2: &[sunrealtype],
    s: i32,
) -> sunbooleantype {
    let mut bAAcAc: sunrealtype = 0.0;
    let mut tmp1 = butcher_tmp(s);
    let mut tmp2 = butcher_tmp(s);
    let mut tmp3 = butcher_tmp(s);
    if arkode_butcher_mv(A3, c2, s, &mut tmp1) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_mv(A2, c1, s, &mut tmp2) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_vv(&tmp1, &tmp2, s, &mut tmp3) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_mv(A1, &tmp3, s, &mut tmp1) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_dot(b, &tmp1, s, &mut bAAcAc) != 0 {
        return SUNFALSE;
    }
    if SUNRabs(bAAcAc - 1.0 / 120.0) > TOL() {
        SUNFALSE
    } else {
        SUNTRUE
    }
}

/* b'*A1*(c1.*(A2*(c2.*c3))) = 1/90 */
fn arkode_butcher_order6n(
    b: &[sunrealtype],
    A1: &[Vec<sunrealtype>],
    c1: &[sunrealtype],
    A2: &[Vec<sunrealtype>],
    c2: &[sunrealtype],
    c3: &[sunrealtype],
    s: i32,
) -> sunbooleantype {
    let mut bAcAcc: sunrealtype = 0.0;
    let mut tmp1 = butcher_tmp(s);
    let mut tmp2 = butcher_tmp(s);
    if arkode_butcher_vv(c2, c3, s, &mut tmp1) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_mv(A2, &tmp1, s, &mut tmp2) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_vv(c1, &tmp2, s, &mut tmp1) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_mv(A1, &tmp1, s, &mut tmp2) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_dot(b, &tmp2, s, &mut bAcAcc) != 0 {
        return SUNFALSE;
    }
    if SUNRabs(bAcAcc - 1.0 / 90.0) > TOL() {
        SUNFALSE
    } else {
        SUNTRUE
    }
}

/* b'*A1*(c1.*(A2*A3*c2)) = 1/180 */
fn arkode_butcher_order6o(
    b: &[sunrealtype],
    A1: &[Vec<sunrealtype>],
    c1: &[sunrealtype],
    A2: &[Vec<sunrealtype>],
    A3: &[Vec<sunrealtype>],
    c2: &[sunrealtype],
    s: i32,
) -> sunbooleantype {
    let mut bAcAAc: sunrealtype = 0.0;
    let mut tmp1 = butcher_tmp(s);
    let mut tmp2 = butcher_tmp(s);
    if arkode_butcher_mv(A3, c2, s, &mut tmp1) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_mv(A2, &tmp1, s, &mut tmp2) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_vv(c1, &tmp2, s, &mut tmp1) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_mv(A1, &tmp1, s, &mut tmp2) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_dot(b, &tmp2, s, &mut bAcAAc) != 0 {
        return SUNFALSE;
    }
    if SUNRabs(bAcAAc - 1.0 / 180.0) > TOL() {
        SUNFALSE
    } else {
        SUNTRUE
    }
}

/* b'*A1*A2*(c1.*c2.*c3) = 1/120 */
fn arkode_butcher_order6p(
    b: &[sunrealtype],
    A1: &[Vec<sunrealtype>],
    A2: &[Vec<sunrealtype>],
    c1: &[sunrealtype],
    c2: &[sunrealtype],
    c3: &[sunrealtype],
    s: i32,
) -> sunbooleantype {
    let mut bAAccc: sunrealtype = 0.0;
    let mut tmp1 = butcher_tmp(s);
    let mut tmp2 = butcher_tmp(s);
    if arkode_butcher_vv(c1, c2, s, &mut tmp1) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_vv(c3, &tmp1, s, &mut tmp2) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_mv(A2, &tmp2, s, &mut tmp1) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_mv(A1, &tmp1, s, &mut tmp2) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_dot(b, &tmp2, s, &mut bAAccc) != 0 {
        return SUNFALSE;
    }
    if SUNRabs(bAAccc - 1.0 / 120.0) > TOL() {
        SUNFALSE
    } else {
        SUNTRUE
    }
}

/* b'*A1*A2*(c1.*(A3*c2)) = 1/240 */
fn arkode_butcher_order6q(
    b: &[sunrealtype],
    A1: &[Vec<sunrealtype>],
    A2: &[Vec<sunrealtype>],
    c1: &[sunrealtype],
    A3: &[Vec<sunrealtype>],
    c2: &[sunrealtype],
    s: i32,
) -> sunbooleantype {
    let mut bAAcAc: sunrealtype = 0.0;
    let mut tmp1 = butcher_tmp(s);
    let mut tmp2 = butcher_tmp(s);
    if arkode_butcher_mv(A3, c2, s, &mut tmp1) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_vv(c1, &tmp1, s, &mut tmp2) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_mv(A2, &tmp2, s, &mut tmp1) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_mv(A1, &tmp1, s, &mut tmp2) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_dot(b, &tmp2, s, &mut bAAcAc) != 0 {
        return SUNFALSE;
    }
    if SUNRabs(bAAcAc - 1.0 / 240.0) > TOL() {
        SUNFALSE
    } else {
        SUNTRUE
    }
}

/* b'*A1*A2*A3*(c1.*c2) = 1/360 */
fn arkode_butcher_order6r(
    b: &[sunrealtype],
    A1: &[Vec<sunrealtype>],
    A2: &[Vec<sunrealtype>],
    A3: &[Vec<sunrealtype>],
    c1: &[sunrealtype],
    c2: &[sunrealtype],
    s: i32,
) -> sunbooleantype {
    let mut bAAAcc: sunrealtype = 0.0;
    let mut tmp1 = butcher_tmp(s);
    let mut tmp2 = butcher_tmp(s);
    if arkode_butcher_vv(c1, c2, s, &mut tmp1) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_mv(A3, &tmp1, s, &mut tmp2) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_mv(A2, &tmp2, s, &mut tmp1) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_mv(A1, &tmp1, s, &mut tmp2) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_dot(b, &tmp2, s, &mut bAAAcc) != 0 {
        return SUNFALSE;
    }
    if SUNRabs(bAAAcc - 1.0 / 360.0) > TOL() {
        SUNFALSE
    } else {
        SUNTRUE
    }
}

/* b'*A1*A2*A3*A4*c = 1/720 */
fn arkode_butcher_order6s(
    b: &[sunrealtype],
    A1: &[Vec<sunrealtype>],
    A2: &[Vec<sunrealtype>],
    A3: &[Vec<sunrealtype>],
    A4: &[Vec<sunrealtype>],
    c: &[sunrealtype],
    s: i32,
) -> sunbooleantype {
    let mut bAAAAc: sunrealtype = 0.0;
    let mut tmp1 = butcher_tmp(s);
    let mut tmp2 = butcher_tmp(s);
    if arkode_butcher_mv(A4, c, s, &mut tmp1) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_mv(A3, &tmp1, s, &mut tmp2) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_mv(A2, &tmp2, s, &mut tmp1) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_mv(A1, &tmp1, s, &mut tmp2) != 0 {
        return SUNFALSE;
    }
    if arkode_butcher_dot(b, &tmp2, s, &mut bAAAAc) != 0 {
        return SUNFALSE;
    }
    if SUNRabs(bAAAAc - 1.0 / 720.0) > TOL() {
        SUNFALSE
    } else {
        SUNTRUE
    }
}

/*---------------------------------------------------------------
  Utility routine to check Butcher's simplifying assumptions.
  Returns the maximum predicted order.
  ---------------------------------------------------------------*/
fn __ButcherSimplifyingAssumptions(
    A: &[Vec<sunrealtype>],
    b: &[sunrealtype],
    c: &[sunrealtype],
    s: i32,
) -> i32 {
    let mut tmp = butcher_tmp(s);

    /* B(P) */
    let mut P: i32 = 0;
    for i in 1..1000 {
        if arkode_butcher_vp(c, i - 1, s, &mut tmp) != 0 {
            return 0;
        }
        let mut LHS: sunrealtype = 0.0;
        if arkode_butcher_dot(b, &tmp, s, &mut LHS) != 0 {
            return 0;
        }
        let RHS: sunrealtype = 1.0 / (i as sunrealtype);
        if SUNRabs(RHS - LHS) > TOL() {
            break;
        }
        P += 1;
    }

    /* C(Q) */
    let mut Q: i32 = 0;
    for k in 1..1000 {
        let mut alltrue: sunbooleantype = SUNTRUE;
        for i in 0..s {
            if arkode_butcher_vp(c, k - 1, s, &mut tmp) != 0 {
                return 0;
            }
            let mut LHS: sunrealtype = 0.0;
            if arkode_butcher_dot(&A[i as usize], &tmp, s, &mut LHS) != 0 {
                return 0;
            }
            let RHS: sunrealtype = SUNRpowerI(c[i as usize], k) / (k as sunrealtype);
            if SUNRabs(RHS - LHS) > TOL() {
                alltrue = SUNFALSE;
                break;
            }
        }
        if alltrue {
            Q += 1;
        } else {
            break;
        }
    }

    /* D(R) */
    let mut R: i32 = 0;
    for k in 1..1000 {
        let mut alltrue: sunbooleantype = SUNTRUE;
        for j in 0..s {
            let mut LHS: sunrealtype = 0.0;
            for i in 0..s {
                LHS += A[i as usize][j as usize]
                    * b[i as usize]
                    * SUNRpowerI(c[i as usize], k - 1);
            }
            let RHS: sunrealtype =
                b[j as usize] / (k as sunrealtype) * (1.0 - SUNRpowerI(c[j as usize], k));
            if SUNRabs(RHS - LHS) > TOL() {
                alltrue = SUNFALSE;
                break;
            }
        }
        if alltrue {
            R += 1;
        } else {
            break;
        }
    }

    /* determine q, clean up and return */
    let mut q: i32 = 0;
    for _i in 1..=P {
        if (q > Q + R + 1) || (q > 2 * Q + 2) {
            break;
        }
        q += 1;
    }
    q
}

/*---------------------------------------------------------------
  EOF
  ---------------------------------------------------------------*/
