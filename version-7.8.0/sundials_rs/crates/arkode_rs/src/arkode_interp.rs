//! Port of `src/arkode/arkode_interp.c` + `src/arkode/arkode_interp_impl.h`
//! (ARKODE's temporal interpolation utility: the generic dispatchers plus the
//! Hermite and Lagrange implementations).
//!
//! `_generic_ARKInterp` / `_generic_ARKInterpOps` / `ARKInterp` / `ARKInterpOps`
//! live in the frozen contract (`arkode_impl.rs`, section 6); this module owns
//! both concrete contents, their downcast helpers, the two constructors, and
//! the seven dispatchers.
//!
//! Binding notes:
//! * The `HINT_*` / `LINT_*` accessor macros become the two `content_mut`
//!   downcast helpers (`hermite_content_mut`, `lagrange_content_mut`) returning
//!   `RefMut` guards. A guard is never held across `arkAllocVec` /
//!   `arkFreeVec` / `arkResizeVec` / `arkProcessError` / a `step_fullrhs` call
//!   / the recursive `arkInterpEvaluate` (all of which borrow `ark_mem`, and
//!   the recursion re-borrows this very content): every such site copies the
//!   fields it needs into locals, drops the guard, then calls out. Where C
//!   passes `&HINT_FOLD(interp)` into `arkAllocVec`/`arkResizeVec` so the
//!   callee can replace the vector, the port `Option::take`s the field, calls
//!   with `&mut` on the local, and writes it back on EVERY path.
//! * `interp == NULL` -> `Option<&ARKInterp>` on the dispatchers (the C
//!   NULL test); the ops themselves receive a non-null `&ARKInterp`, as in the
//!   contract's ops table.
//! * `interp->content == NULL` (after a free) is `Box::new(())` in
//!   `RefCell<Box<dyn Any>>`, tested with `is::<()>()`. `free(interp->ops)` /
//!   `free(interp)` cannot be expressed for an `Rc` handle: the ops table is
//!   reset to `Default` and the storage is released when the last handle is
//!   dropped, so a double `arkInterpFree` is a no-op here instead of C UB.
//! * C `N_Vector*`/`sunrealtype*` history arrays become `Vec<Option<N_Vector>>`
//!   / `Vec<sunrealtype>`; the C `array == NULL` state is the empty `Vec`.
//! * `malloc` failure branches are unreachable and dropped; `arkAllocVec` /
//!   `arkResizeVec` failures (which are reachable -- a vector missing
//!   `nvclone`/`nvdestroy`) keep their C handling.
//! * `SUNLogDebug` compiles away at `SUNDIALS_LOGGING_LEVEL=2`;
//!   `SUNDIALS_DEBUG_PRINTVEC` is not defined in the reference build, so the
//!   `N_VPrintFile` blocks are omitted.
//!
//! Accepted deviation (class 5, unobservable): `arkInterpPrintMem_Lagrange`
//! cannot reproduce C's `%p` rendering of the `yhist` pointers and prints a
//! fixed placeholder instead. No reference example calls `ARKodePrintMem`.

use std::any::Any;
use std::cell::{RefCell, RefMut};
use std::rc::Rc;

use crate::arkode::{arkAllocVec, arkFreeVec, arkResizeVec};
use crate::arkode_impl::*;
use sundials_core::sundials_math::{SUNRabs, SUNMAX, SUNMIN};
use sundials_core::sundials_nvector::{
    N_VConst, N_VConstVectorArray, N_VLinearCombination, N_VLinearSum, N_VScale, N_Vector,
};
use sundials_core::sundials_types::*;
use sundials_core::sundials_utils::{sun_format_g, SUNFile};

/*===============================================================
  ARKODE temporal interpolation constants (arkode_interp_impl.h).
  `THREE` is shared with the rest of ARKODE and lives in the
  contract; the remaining three are module-local, as prescribed.
  ===============================================================*/

pub const FOURTH: sunrealtype = 0.25;
pub const SIX: sunrealtype = 6.0;
pub const TWELVE: sunrealtype = 12.0;

/*---------------------------------------------------------------
  Section I: generic ARKInterp functions provided by all
  interpolation modules
  ---------------------------------------------------------------*/

pub fn arkInterpResize(
    ark_mem: &ARKodeMem,
    interp: Option<&ARKInterp>,
    resize: Option<ARKVecResizeFn>,
    resize_data: &mut Option<Box<dyn Any>>,
    lrw_diff: sunindextype,
    liw_diff: sunindextype,
    tmpl: &N_Vector,
) -> i32 {
    let interp = match interp {
        None => return ARK_SUCCESS,
        Some(interp) => interp,
    };
    let op = interp.ops.borrow().resize.expect("interp ops resize");
    op(
        ark_mem,
        interp,
        resize,
        resize_data,
        lrw_diff,
        liw_diff,
        tmpl,
    )
}

pub fn arkInterpFree(ark_mem: &ARKodeMem, interp: Option<&ARKInterp>) {
    let interp = match interp {
        None => return,
        Some(interp) => interp,
    };
    let op = interp.ops.borrow().free.expect("interp ops free");
    op(ark_mem, interp);
}

pub fn arkInterpPrintMem(interp: Option<&ARKInterp>, outfile: &SUNFile) {
    let interp = match interp {
        None => return,
        Some(interp) => interp,
    };
    let op = interp.ops.borrow().print.expect("interp ops print");
    op(interp, outfile);
}

pub fn arkInterpSetDegree(ark_mem: &ARKodeMem, interp: Option<&ARKInterp>, degree: i32) -> i32 {
    let interp = match interp {
        None => return ARK_SUCCESS,
        Some(interp) => interp,
    };
    let op = interp.ops.borrow().setdegree.expect("interp ops setdegree");
    op(ark_mem, interp, degree)
}

pub fn arkInterpInit(ark_mem: &ARKodeMem, interp: Option<&ARKInterp>, tnew: sunrealtype) -> i32 {
    let interp = match interp {
        None => return ARK_SUCCESS,
        Some(interp) => interp,
    };
    let op = interp.ops.borrow().init.expect("interp ops init");
    op(ark_mem, interp, tnew)
}

pub fn arkInterpUpdate(ark_mem: &ARKodeMem, interp: Option<&ARKInterp>, tnew: sunrealtype) -> i32 {
    let interp = match interp {
        None => return ARK_SUCCESS,
        Some(interp) => interp,
    };
    let op = interp.ops.borrow().update.expect("interp ops update");
    op(ark_mem, interp, tnew)
}

pub fn arkInterpEvaluate(
    ark_mem: &ARKodeMem,
    interp: Option<&ARKInterp>,
    tau: sunrealtype,
    d: i32,
    order: i32,
    yout: &N_Vector,
) -> i32 {
    let interp = match interp {
        None => return ARK_SUCCESS,
        Some(interp) => interp,
    };
    let op = interp.ops.borrow().evaluate.expect("interp ops evaluate");
    op(ark_mem, interp, tau, d, order, yout)
}

/*===============================================================
  ARKODE Hermite Temporal Interpolation Data Structure
  ===============================================================*/

/* Hermite interpolation structure */
pub struct _ARKInterpContent_Hermite {
    pub degree: i32,               /* maximum interpolant degree to use           */
    pub fold: Option<N_Vector>,    /* f(t,y) at beginning of last successful step */
    pub yold: Option<N_Vector>,    /* y at beginning of last successful step      */
    pub fa: Option<N_Vector>,      /* f(t,y) used in higher-order interpolation   */
    pub fb: Option<N_Vector>,      /* f(t,y) used in higher-order interpolation   */
    pub told: sunrealtype,         /* t at beginning of last successful step      */
    pub tnew: sunrealtype,         /* t at end of last successful step            */
    pub h: sunrealtype,            /* last successful step size                   */
}

pub type ARKInterpContent_Hermite = _ARKInterpContent_Hermite;

/// Hermite structure accessor (C macros `HINT_CONTENT`/`HINT_DEGREE`/...).
/// Panics if the interpolation module is not the Hermite one, or if its
/// content has already been freed (C would blindly cast the `void*` -- UB maps
/// to a panic, deviation class 5). NEVER hold the guard across a call that
/// borrows `ark_mem` or re-enters this module.
pub fn hermite_content_mut(interp: &ARKInterp) -> RefMut<'_, _ARKInterpContent_Hermite> {
    RefMut::map(interp.content.borrow_mut(), |c| {
        c.downcast_mut::<_ARKInterpContent_Hermite>()
            .expect("Hermite ARKInterp content")
    })
}

/*---------------------------------------------------------------
  Section II: Hermite interpolation module implementation
  ---------------------------------------------------------------*/

/*---------------------------------------------------------------
  arkInterpCreate_Hermite:

  This routine creates an ARKInterp structure, through
  cloning an input template N_Vector.  This returns a non-NULL
  structure if no errors occurred, or a NULL value otherwise.
  ---------------------------------------------------------------*/
pub fn arkInterpCreate_Hermite(ark_mem: &ARKodeMem, degree: i32) -> Option<ARKInterp> {
    /* check for valid degree */
    if degree < 0 || degree > ARK_INTERP_MAX_DEGREE {
        return None;
    }

    /* allocate overall structure (C `malloc` cannot fail here) */

    /* allocate ops structure and set entries */
    let ops = ARKInterpOps {
        resize: Some(arkInterpResize_Hermite),
        free: Some(arkInterpFree_Hermite),
        print: Some(arkInterpPrintMem_Hermite),
        setdegree: Some(arkInterpSetDegree_Hermite),
        init: Some(arkInterpInit_Hermite),
        update: Some(arkInterpUpdate_Hermite),
        evaluate: Some(arkInterpEvaluate_Hermite),
    };

    /* create content, and initialize everything to zero/NULL */
    let content = _ARKInterpContent_Hermite {
        degree: 0,
        fold: None,
        yold: None,
        fa: None,
        fb: None,
        told: ZERO,
        tnew: ZERO,
        h: ZERO,
    };

    /* attach ops and content structures to overall structure */
    let interp: ARKInterp = Rc::new(_generic_ARKInterp {
        content: RefCell::new(Box::new(content)),
        ops: RefCell::new(ops),
    });

    /* fill content */
    {
        let mut c = hermite_content_mut(&interp);

        /* initialize local N_Vectors to NULL */
        c.fold = None;
        c.yold = None;
        c.fa = None;
        c.fb = None;

        /* set maximum interpolant degree */
        c.degree = SUNMIN(ARK_INTERP_MAX_DEGREE, degree);
    }

    /* update workspace sizes */
    {
        let mut m = ark_mem.borrow_mut();
        m.lrw += 2;
        m.liw += 5;
    }

    /* initialize time values */
    {
        let tcur = ark_mem.borrow().tcur;
        let mut c = hermite_content_mut(&interp);
        c.told = tcur;
        c.tnew = tcur;
        c.h = 0.0;
    }

    Some(interp)
}

/*---------------------------------------------------------------
  arkInterpResize_Hermite:

  This routine resizes the internal vectors.
  ---------------------------------------------------------------*/
pub fn arkInterpResize_Hermite(
    ark_mem: &ARKodeMem,
    interp: &ARKInterp,
    resize: Option<ARKVecResizeFn>,
    resize_data: &mut Option<Box<dyn Any>>,
    lrw_diff: sunindextype,
    liw_diff: sunindextype,
    y0: &N_Vector,
) -> i32 {
    /* resize vectors (the C `interp == NULL` test is unrepresentable here) */

    {
        let mut v = hermite_content_mut(interp).fold.take();
        let ok = arkResizeVec(ark_mem, resize, resize_data, lrw_diff, liw_diff, y0, &mut v);
        hermite_content_mut(interp).fold = v;
        if !ok {
            return ARK_MEM_FAIL;
        }
    }

    {
        let mut v = hermite_content_mut(interp).yold.take();
        let ok = arkResizeVec(ark_mem, resize, resize_data, lrw_diff, liw_diff, y0, &mut v);
        hermite_content_mut(interp).yold = v;
        if !ok {
            return ARK_MEM_FAIL;
        }
    }

    {
        let mut v = hermite_content_mut(interp).fa.take();
        let ok = arkResizeVec(ark_mem, resize, resize_data, lrw_diff, liw_diff, y0, &mut v);
        hermite_content_mut(interp).fa = v;
        if !ok {
            return ARK_MEM_FAIL;
        }
    }

    {
        let mut v = hermite_content_mut(interp).fb.take();
        let ok = arkResizeVec(ark_mem, resize, resize_data, lrw_diff, liw_diff, y0, &mut v);
        hermite_content_mut(interp).fb = v;
        if !ok {
            return ARK_MEM_FAIL;
        }
    }

    /* reinitialize time values */
    {
        let tcur = ark_mem.borrow().tcur;
        let mut c = hermite_content_mut(interp);
        c.told = tcur;
        c.tnew = tcur;
        c.h = 0.0;
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  arkInterpFree_Hermite:

  This routine frees the Hermite ARKInterp structure.
  ---------------------------------------------------------------*/
pub fn arkInterpFree_Hermite(ark_mem: &ARKodeMem, interp: &ARKInterp) {
    /* if interpolation structure is NULL, just return -- unrepresentable */

    /* free content (C `interp->content != NULL`) */
    let has_content = interp.content.borrow().downcast_ref::<()>().is_none();
    if has_content {
        {
            let mut v = hermite_content_mut(interp).fold.take();
            if v.is_some() {
                arkFreeVec(ark_mem, &mut v);
            }
            hermite_content_mut(interp).fold = v;
        }
        {
            let mut v = hermite_content_mut(interp).yold.take();
            if v.is_some() {
                arkFreeVec(ark_mem, &mut v);
            }
            hermite_content_mut(interp).yold = v;
        }
        {
            let mut v = hermite_content_mut(interp).fa.take();
            if v.is_some() {
                arkFreeVec(ark_mem, &mut v);
            }
            hermite_content_mut(interp).fa = v;
        }
        {
            let mut v = hermite_content_mut(interp).fb.take();
            if v.is_some() {
                arkFreeVec(ark_mem, &mut v);
            }
            hermite_content_mut(interp).fb = v;
        }

        /* update work space sizes */
        {
            let mut m = ark_mem.borrow_mut();
            m.lrw -= 2;
            m.liw -= 5;
        }

        /* C: free(interp->content); interp->content = NULL */
        *interp.content.borrow_mut() = Box::new(());
    }

    /* free ops and interpolation structures: C `free(interp->ops)` /
    `free(interp)`; the handle's storage goes when the last `Rc` does */
    *interp.ops.borrow_mut() = ARKInterpOps::default();
}

/*---------------------------------------------------------------
  arkInterpPrintMem_Hermite

  This routine outputs the Hermite temporal interpolation memory
  structure to a specified file pointer.
  ---------------------------------------------------------------*/
pub fn arkInterpPrintMem_Hermite(interp: &ARKInterp, outfile: &SUNFile) {
    /* the C `interp != NULL` test is unrepresentable here */
    let (degree, told, tnew, h) = {
        let c = hermite_content_mut(interp);
        (c.degree, c.told, c.tnew, c.h)
    };
    outfile.write_str(&format!("arkode_interp (Hermite): degree = {}\n", degree));
    outfile.write_str(&format!(
        "arkode_interp (Hermite): told = {}\n",
        sun_format_g(told)
    ));
    outfile.write_str(&format!(
        "arkode_interp (Hermite): tnew = {}\n",
        sun_format_g(tnew)
    ));
    outfile.write_str(&format!(
        "arkode_interp (Hermite): h = {}\n",
        sun_format_g(h)
    ));
    /* SUNDIALS_DEBUG_PRINTVEC is not defined in the reference build */
}

/*---------------------------------------------------------------
  arkInterpSetDegree_Hermite

  This routine sets a supplied interpolation degree which must be
  in the range 0 <= degree <= ARK_INTERP_MAX_DEGREE.

  Return values:
    ARK_ILL_INPUT -- if the input is outside of allowable bounds
    ARK_INTERP_FAIL -- if the interpolation module has already
       been initialized,
    ARK_SUCCESS -- successful completion.
  ---------------------------------------------------------------*/
pub fn arkInterpSetDegree_Hermite(ark_mem: &ARKodeMem, interp: &ARKInterp, degree: i32) -> i32 {
    if degree > ARK_INTERP_MAX_DEGREE || degree < 0 {
        arkProcessError(
            Some(ark_mem),
            ARK_INTERP_FAIL,
            line!() as i32,
            "arkInterpSetDegree_Hermite",
            file!(),
            "Illegal degree specified.",
        );
        return ARK_ILL_INPUT;
    }

    hermite_content_mut(interp).degree = degree;

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  arkInterpInit_Hermite

  This routine performs the following steps:
  1. Sets tnew and told to the input time
  2. Allocates any missing/needed N_Vector storage (for reinit)
  3. Copies ark_mem->yn into yold
  4. Calls the full RHS routine to fill fnew
  5. Copies fnew into fold
  ---------------------------------------------------------------*/
pub fn arkInterpInit_Hermite(ark_mem: &ARKodeMem, interp: &ARKInterp, tnew: sunrealtype) -> i32 {
    /* initialize time values */
    {
        let mut c = hermite_content_mut(interp);
        c.told = tnew;
        c.tnew = tnew;
        c.h = 0.0;
    }

    /* allocate vectors based on interpolant degree */
    if hermite_content_mut(interp).fold.is_none() {
        let yn = ark_mem.borrow().yn.clone().expect("yn");
        let mut v = hermite_content_mut(interp).fold.take();
        let ok = arkAllocVec(ark_mem, &yn, &mut v);
        hermite_content_mut(interp).fold = v;
        if !ok {
            arkInterpFree(ark_mem, Some(interp));
            return ARK_MEM_FAIL;
        }
    }
    if hermite_content_mut(interp).yold.is_none() {
        let yn = ark_mem.borrow().yn.clone().expect("yn");
        let mut v = hermite_content_mut(interp).yold.take();
        let ok = arkAllocVec(ark_mem, &yn, &mut v);
        hermite_content_mut(interp).yold = v;
        if !ok {
            arkInterpFree(ark_mem, Some(interp));
            return ARK_MEM_FAIL;
        }
    }
    {
        let need = {
            let c = hermite_content_mut(interp);
            (c.degree > 3) && c.fa.is_none()
        };
        if need {
            let yn = ark_mem.borrow().yn.clone().expect("yn");
            let mut v = hermite_content_mut(interp).fa.take();
            let ok = arkAllocVec(ark_mem, &yn, &mut v);
            hermite_content_mut(interp).fa = v;
            if !ok {
                arkInterpFree(ark_mem, Some(interp));
                return ARK_MEM_FAIL;
            }
        }
    }
    {
        let need = {
            let c = hermite_content_mut(interp);
            (c.degree > 4) && c.fb.is_none()
        };
        if need {
            let yn = ark_mem.borrow().yn.clone().expect("yn");
            let mut v = hermite_content_mut(interp).fb.take();
            let ok = arkAllocVec(ark_mem, &yn, &mut v);
            hermite_content_mut(interp).fb = v;
            if !ok {
                arkInterpFree(ark_mem, Some(interp));
                return ARK_MEM_FAIL;
            }
        }
    }

    /* signal that a full RHS data is required for interpolation */
    ark_mem.borrow_mut().call_fullrhs = SUNTRUE;

    /* return with success */
    ARK_SUCCESS
}

/*---------------------------------------------------------------
  arkInterpUpdate_Hermite

  This routine copies ynew into yold, and fnew into fold, so that
  yold and fold contain the previous values.
  ---------------------------------------------------------------*/
pub fn arkInterpUpdate_Hermite(ark_mem: &ARKodeMem, interp: &ARKInterp, tnew: sunrealtype) -> i32 {
    /* call full RHS if needed -- called just BEFORE the end of a step, so yn has
       NOT been updated to ycur yet */
    if !ark_mem.borrow().fn_is_current {
        let (step_fullrhs, tn, yn, fn_) = {
            let m = ark_mem.borrow();
            (
                m.step_fullrhs.expect("step_fullrhs"),
                m.tn,
                m.yn.clone().expect("yn"),
                m.fn_.clone().expect("fn"),
            )
        };
        let retval = step_fullrhs(ark_mem, tn, &yn, &fn_, ARK_FULLRHS_START);
        if retval != 0 {
            return ARK_RHSFUNC_FAIL;
        }
        ark_mem.borrow_mut().fn_is_current = SUNTRUE;
    }

    /* copy ynew and fnew into yold and fold, respectively */
    let (yn, fn_) = {
        let m = ark_mem.borrow();
        (m.yn.clone().expect("yn"), m.fn_.clone().expect("fn"))
    };
    let (yold, fold) = {
        let c = hermite_content_mut(interp);
        (
            c.yold.clone().expect("interp yold"),
            c.fold.clone().expect("interp fold"),
        )
    };
    N_VScale(ONE, &yn, &yold);
    N_VScale(ONE, &fn_, &fold);

    /* update time values */
    {
        let h = ark_mem.borrow().h;
        let mut c = hermite_content_mut(interp);
        c.told = c.tnew;
        c.tnew = tnew;
        c.h = h;
    }

    /* return with success */
    ARK_SUCCESS
}

/*---------------------------------------------------------------
  arkInterpEvaluate_Hermite

  This routine evaluates a temporal interpolation/extrapolation
  based on the data in the interpolation structure:
     yold = y(told)
     ynew = y(tnew)
     fold = f(told, yold)
     fnew = f(told, ynew)
  This typically consists of using a cubic Hermite interpolating
  formula with this data.  If greater polynomial degree than 3 is
  requested, then we can bootstrap up to a 5th-order interpolant.
  For lower order interpolants than cubic, we use:
     {yold,ynew,fnew} for quadratic
     {yold,ynew} for linear
     {0.5*(yold+ynew)} for constant.

  Derivatives have lower accuracy than the interpolant
  itself, losing one order per derivative.  We will provide
  derivatives up to d = min(5,q).

  The input 'tau' specifies the time at which to evaluate the Hermite
  polynomial.  The formula for tau is defined using the
  most-recently-completed solution interval [told,tnew], and is
  given by:
               t = tnew + tau*(tnew-told),
  where h = tnew-told, i.e. values -1<tau<0 provide interpolation,
  other values result in extrapolation.
  ---------------------------------------------------------------*/
pub fn arkInterpEvaluate_Hermite(
    ark_mem: &ARKodeMem,
    interp: &ARKInterp,
    tau: sunrealtype,
    d: i32,
    order: i32,
    yout: &N_Vector,
) -> i32 {
    /* local variables */
    let mut a: [sunrealtype; 6] = [ZERO; 6];

    /* set constants */
    let tau2 = tau * tau;
    let tau3 = tau * tau2;
    let tau4 = tau * tau3;
    let tau5 = tau * tau4;

    let (h, degree) = {
        let c = hermite_content_mut(interp);
        (c.h, c.degree)
    };
    let h2 = h * h;
    let h3 = h * h2;
    let h4 = h * h3;
    let h5 = h * h4;

    /* determine polynomial order q */
    let mut q = SUNMAX(order, 0); /* respect lower bound  */
    q = SUNMIN(q, degree); /* respect max possible */

    /* call full RHS if needed -- called just AFTER the end of a step, so yn has
       been updated to ycur */
    if !ark_mem.borrow().fn_is_current {
        let (step_fullrhs, tn, yn, fn_) = {
            let m = ark_mem.borrow();
            (
                m.step_fullrhs.expect("step_fullrhs"),
                m.tn,
                m.yn.clone().expect("yn"),
                m.fn_.clone().expect("fn"),
            )
        };
        let retval = step_fullrhs(ark_mem, tn, &yn, &fn_, ARK_FULLRHS_END);
        if retval != 0 {
            return ARK_RHSFUNC_FAIL;
        }
        ark_mem.borrow_mut().fn_is_current = SUNTRUE;
    }

    /* error on illegal d */
    if d < 0 {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "arkInterpEvaluate_Hermite",
            file!(),
            "Requested illegal derivative.",
        );
        return ARK_ILL_INPUT;
    }

    /* if d is too high, just return zeros */
    if d > q {
        N_VConst(ZERO, yout);
        return ARK_SUCCESS;
    }

    /* build polynomial based on order */
    match q {
        0 => {
            /* constant interpolant, yout = 0.5*(yn+yp) */
            let yold = hermite_content_mut(interp).yold.clone().expect("interp yold");
            let yn = ark_mem.borrow().yn.clone().expect("yn");
            N_VLinearSum(HALF, &yold, HALF, &yn, yout);
        }

        1 => {
            /* linear interpolant */
            let a0;
            let a1;
            if d == 0 {
                a0 = -tau;
                a1 = ONE + tau;
            } else {
                /* d=1 */
                a0 = -ONE / h;
                a1 = ONE / h;
            }
            let yold = hermite_content_mut(interp).yold.clone().expect("interp yold");
            let yn = ark_mem.borrow().yn.clone().expect("yn");
            N_VLinearSum(a0, &yold, a1, &yn, yout);
        }

        2 => {
            /* quadratic interpolant */
            if d == 0 {
                a[0] = tau2;
                a[1] = ONE - tau2;
                a[2] = h * (tau2 + tau);
            } else if d == 1 {
                a[0] = TWO * tau / h;
                a[1] = -TWO * tau / h;
                a[2] = ONE + TWO * tau;
            } else {
                /* d == 2 */
                a[0] = TWO / h / h;
                a[1] = -TWO / h / h;
                a[2] = TWO / h;
            }
            let yold = hermite_content_mut(interp).yold.clone().expect("interp yold");
            let (yn, fn_) = {
                let m = ark_mem.borrow();
                (m.yn.clone().expect("yn"), m.fn_.clone().expect("fn"))
            };
            let X: [N_Vector; 3] = [yold, yn, fn_];
            let retval = N_VLinearCombination(3, &a[..3], &X, yout);
            if retval != 0 {
                return ARK_VECTOROP_ERR;
            }
        }

        3 => {
            /* cubic interpolant */
            if d == 0 {
                a[0] = THREE * tau2 + TWO * tau3;
                a[1] = ONE - THREE * tau2 - TWO * tau3;
                a[2] = h * (tau2 + tau3);
                a[3] = h * (tau + TWO * tau2 + tau3);
            } else if d == 1 {
                a[0] = SIX * (tau + tau2) / h;
                a[1] = -SIX * (tau + tau2) / h;
                a[2] = TWO * tau + THREE * tau2;
                a[3] = ONE + FOUR * tau + THREE * tau2;
            } else if d == 2 {
                a[0] = SIX * (ONE + TWO * tau) / h2;
                a[1] = -SIX * (ONE + TWO * tau) / h2;
                a[2] = (TWO + SIX * tau) / h;
                a[3] = (FOUR + SIX * tau) / h;
            } else {
                /* d == 3 */
                a[0] = TWELVE / h3;
                a[1] = -TWELVE / h3;
                a[2] = SIX / h2;
                a[3] = SIX / h2;
            }
            let (yold, fold) = {
                let c = hermite_content_mut(interp);
                (
                    c.yold.clone().expect("interp yold"),
                    c.fold.clone().expect("interp fold"),
                )
            };
            let (yn, fn_) = {
                let m = ark_mem.borrow();
                (m.yn.clone().expect("yn"), m.fn_.clone().expect("fn"))
            };
            let X: [N_Vector; 4] = [yold, yn, fold, fn_];
            let retval = N_VLinearCombination(4, &a[..4], &X, yout);
            if retval != 0 {
                return ARK_VECTOROP_ERR;
            }
        }

        4 => {
            /* quartic interpolant */

            /* first, evaluate cubic interpolant at tau=-1/3 */
            let mut tval = -ONE / THREE;
            let retval = arkInterpEvaluate(ark_mem, Some(interp), tval, 0, 3, yout);
            if retval != 0 {
                return ARK_RHSFUNC_FAIL;
            }

            /* second, evaluate RHS at tau=-1/3, storing the result in fa */
            tval = hermite_content_mut(interp).tnew - h / THREE;
            let fa = hermite_content_mut(interp).fa.clone().expect("interp fa");
            let step_fullrhs = ark_mem.borrow().step_fullrhs.expect("step_fullrhs");
            let retval = step_fullrhs(ark_mem, tval, yout, &fa, ARK_FULLRHS_OTHER);
            if retval != 0 {
                return ARK_RHSFUNC_FAIL;
            }

            /* evaluate desired function */
            if d == 0 {
                a[0] = -SIX * tau2 - 16.0 * tau3 - 9.0 * tau4;
                a[1] = ONE + SIX * tau2 + 16.0 * tau3 + 9.0 * tau4;
                a[2] = h * FOURTH * (-FIVE * tau2 - 14.0 * tau3 - 9.0 * tau4);
                a[3] = h * (tau + TWO * tau2 + tau3);
                a[4] = h * 27.0 * FOURTH * (-tau4 - TWO * tau3 - tau2);
            } else if d == 1 {
                a[0] = (-TWELVE * tau - 48.0 * tau2 - 36.0 * tau3) / h;
                a[1] = (TWELVE * tau + 48.0 * tau2 + 36.0 * tau3) / h;
                a[2] = HALF * (-FIVE * tau - 21.0 * tau2 - 18.0 * tau3);
                a[3] = ONE + FOUR * tau + THREE * tau2;
                a[4] = -27.0 * HALF * (TWO * tau3 + THREE * tau2 + tau);
            } else if d == 2 {
                a[0] = (-TWELVE - 96.0 * tau - 108.0 * tau2) / h2;
                a[1] = (TWELVE + 96.0 * tau + 108.0 * tau2) / h2;
                a[2] = (-FIVE * HALF - 21.0 * tau - 27.0 * tau2) / h;
                a[3] = (FOUR + SIX * tau) / h;
                a[4] = (-27.0 * HALF - 81.0 * tau - 81.0 * tau2) / h;
            } else if d == 3 {
                a[0] = (-96.0 - 216.0 * tau) / h3;
                a[1] = (96.0 + 216.0 * tau) / h3;
                a[2] = (-21.0 - 54.0 * tau) / h2;
                a[3] = SIX / h2;
                a[4] = (-81.0 - 162.0 * tau) / h2;
            } else {
                /* d == 4 */
                a[0] = -216.0 / h4;
                a[1] = 216.0 / h4;
                a[2] = -54.0 / h3;
                a[3] = ZERO;
                a[4] = -162.0 / h3;
            }
            let (yold, fold, fa) = {
                let c = hermite_content_mut(interp);
                (
                    c.yold.clone().expect("interp yold"),
                    c.fold.clone().expect("interp fold"),
                    c.fa.clone().expect("interp fa"),
                )
            };
            let (yn, fn_) = {
                let m = ark_mem.borrow();
                (m.yn.clone().expect("yn"), m.fn_.clone().expect("fn"))
            };
            let X: [N_Vector; 5] = [yold, yn, fold, fn_, fa];
            let retval = N_VLinearCombination(5, &a[..5], &X, yout);
            if retval != 0 {
                return ARK_VECTOROP_ERR;
            }
        }

        5 => {
            /* quintic interpolant */

            /* first, evaluate quartic interpolant at tau=-1/3 */
            let mut tval = -ONE / THREE;
            let retval = arkInterpEvaluate(ark_mem, Some(interp), tval, 0, 4, yout);
            if retval != 0 {
                return ARK_RHSFUNC_FAIL;
            }

            /* second, evaluate RHS at tau=-1/3, storing the result in fa */
            tval = hermite_content_mut(interp).tnew - h / THREE;
            let fa = hermite_content_mut(interp).fa.clone().expect("interp fa");
            let step_fullrhs = ark_mem.borrow().step_fullrhs.expect("step_fullrhs");
            let retval = step_fullrhs(ark_mem, tval, yout, &fa, ARK_FULLRHS_OTHER);
            if retval != 0 {
                return ARK_RHSFUNC_FAIL;
            }

            /* third, evaluate quartic interpolant at tau=-2/3 */
            tval = -TWO / THREE;
            let retval = arkInterpEvaluate(ark_mem, Some(interp), tval, 0, 4, yout);
            if retval != 0 {
                return ARK_RHSFUNC_FAIL;
            }

            /* fourth, evaluate RHS at tau=-2/3, storing the result in fb */
            tval = hermite_content_mut(interp).tnew - h * TWO / THREE;
            let fb = hermite_content_mut(interp).fb.clone().expect("interp fb");
            let step_fullrhs = ark_mem.borrow().step_fullrhs.expect("step_fullrhs");
            let retval = step_fullrhs(ark_mem, tval, yout, &fb, ARK_FULLRHS_OTHER);
            if retval != 0 {
                return ARK_RHSFUNC_FAIL;
            }

            /* evaluate desired function */
            if d == 0 {
                a[0] = 54.0 * tau5 + 135.0 * tau4 + 110.0 * tau3 + 30.0 * tau2;
                a[1] = ONE - a[0];
                a[2] = h / FOUR * (27.0 * tau5 + 63.0 * tau4 + 49.0 * tau3 + 13.0 * tau2);
                a[3] = h / FOUR
                    * (27.0 * tau5 + 72.0 * tau4 + 67.0 * tau3 + 26.0 * tau2 + FOUR * tau);
                a[4] = h / FOUR * (81.0 * tau5 + 189.0 * tau4 + 135.0 * tau3 + 27.0 * tau2);
                a[5] = h / FOUR * (81.0 * tau5 + 216.0 * tau4 + 189.0 * tau3 + 54.0 * tau2);
            } else if d == 1 {
                a[0] = (270.0 * tau4 + 540.0 * tau3 + 330.0 * tau2 + 60.0 * tau) / h;
                a[1] = -a[0];
                a[2] = (135.0 * tau4 + 252.0 * tau3 + 147.0 * tau2 + 26.0 * tau) / FOUR;
                a[3] = (135.0 * tau4 + 288.0 * tau3 + 201.0 * tau2 + 52.0 * tau + FOUR) / FOUR;
                a[4] = (405.0 * tau4 + 4.0 * 189.0 * tau3 + 405.0 * tau2 + 54.0 * tau) / FOUR;
                a[5] = (405.0 * tau4 + 864.0 * tau3 + 567.0 * tau2 + 108.0 * tau) / FOUR;
            } else if d == 2 {
                a[0] = (1080.0 * tau3 + 1620.0 * tau2 + 660.0 * tau + 60.0) / h2;
                a[1] = -a[0];
                a[2] = (270.0 * tau3 + 378.0 * tau2 + 147.0 * tau + 13.0) / (TWO * h);
                a[3] = (270.0 * tau3 + 432.0 * tau2 + 201.0 * tau + 26.0) / (TWO * h);
                a[4] = (810.0 * tau3 + 1134.0 * tau2 + 405.0 * tau + 27.0) / (TWO * h);
                a[5] = (810.0 * tau3 + 1296.0 * tau2 + 567.0 * tau + 54.0) / (TWO * h);
            } else if d == 3 {
                a[0] = (3240.0 * tau2 + 3240.0 * tau + 660.0) / h3;
                a[1] = -a[0];
                a[2] = (810.0 * tau2 + 756.0 * tau + 147.0) / (TWO * h2);
                a[3] = (810.0 * tau2 + 864.0 * tau + 201.0) / (TWO * h2);
                a[4] = (2430.0 * tau2 + 2268.0 * tau + 405.0) / (TWO * h2);
                a[5] = (2430.0 * tau2 + 2592.0 * tau + 567.0) / (TWO * h2);
            } else if d == 4 {
                a[0] = (6480.0 * tau + 3240.0) / h4;
                a[1] = -a[0];
                a[2] = (810.0 * tau + 378.0) / h3;
                a[3] = (810.0 * tau + 432.0) / h3;
                a[4] = (2430.0 * tau + 1134.0) / h3;
                a[5] = (2430.0 * tau + 1296.0) / h3;
            } else {
                /* d == 5 */
                a[0] = 6480.0 / h5;
                a[1] = -a[0];
                a[2] = 810.0 / h4;
                a[3] = a[2];
                a[4] = 2430.0 / h4;
                a[5] = a[4];
            }
            let (yold, fold, fa, fb) = {
                let c = hermite_content_mut(interp);
                (
                    c.yold.clone().expect("interp yold"),
                    c.fold.clone().expect("interp fold"),
                    c.fa.clone().expect("interp fa"),
                    c.fb.clone().expect("interp fb"),
                )
            };
            let (yn, fn_) = {
                let m = ark_mem.borrow();
                (m.yn.clone().expect("yn"), m.fn_.clone().expect("fn"))
            };
            let X: [N_Vector; 6] = [yold, yn, fold, fn_, fa, fb];
            let retval = N_VLinearCombination(6, &a[..6], &X, yout);
            if retval != 0 {
                return ARK_VECTOROP_ERR;
            }
        }

        _ => {
            arkProcessError(
                Some(ark_mem),
                ARK_ILL_INPUT,
                line!() as i32,
                "arkInterpEvaluate_Hermite",
                file!(),
                "Illegal polynomial order",
            );
            return ARK_ILL_INPUT;
        }
    }

    ARK_SUCCESS
}

/*===============================================================
  ARKODE Lagrange Temporal Interpolation Data Structure
  ===============================================================*/

/* Lagrange interpolation structure */
pub struct _ARKInterpContent_Lagrange {
    pub nmax: i32,                    /* number of previous solutions to use      */
    pub nmaxalloc: i32,               /* vectors allocated for previous solutions */
    pub yhist: Vec<Option<N_Vector>>, /* previous solution vectors                */
    pub thist: Vec<sunrealtype>,      /* 't' values associated with yhist         */
    pub nhist: i32,                   /* number of 'active' vectors in yhist      */
    pub tround: sunrealtype,          /* unit roundoff for 't' values             */
}

pub type ARKInterpContent_Lagrange = _ARKInterpContent_Lagrange;

/// Lagrange structure accessor (C macros `LINT_CONTENT`/`LINT_NMAX`/...).
/// Same rules as [`hermite_content_mut`].
pub fn lagrange_content_mut(interp: &ARKInterp) -> RefMut<'_, _ARKInterpContent_Lagrange> {
    RefMut::map(interp.content.borrow_mut(), |c| {
        c.downcast_mut::<_ARKInterpContent_Lagrange>()
            .expect("Lagrange ARKInterp content")
    })
}

/*---------------------------------------------------------------
  Section III: Lagrange interpolation module implementation
  ---------------------------------------------------------------*/

/*---------------------------------------------------------------
  arkInterpCreate_Lagrange:

  This routine creates an ARKInterp structure, through
  cloning an input template N_Vector.  This returns a non-NULL
  structure if no errors occurred, or a NULL value otherwise.
  ---------------------------------------------------------------*/
pub fn arkInterpCreate_Lagrange(ark_mem: &ARKodeMem, degree: i32) -> Option<ARKInterp> {
    /* check for valid degree */
    if degree < 0 || degree > ARK_INTERP_MAX_DEGREE {
        return None;
    }

    /* allocate overall structure (C `malloc` cannot fail here) */

    /* allocate ops structure and set entries */
    let ops = ARKInterpOps {
        resize: Some(arkInterpResize_Lagrange),
        free: Some(arkInterpFree_Lagrange),
        print: Some(arkInterpPrintMem_Lagrange),
        setdegree: Some(arkInterpSetDegree_Lagrange),
        init: Some(arkInterpInit_Lagrange),
        update: Some(arkInterpUpdate_Lagrange),
        evaluate: Some(arkInterpEvaluate_Lagrange),
    };

    /* create content, and initialize everything to zero/NULL */
    let content = _ARKInterpContent_Lagrange {
        nmax: 0,
        nmaxalloc: 0,
        yhist: Vec::new(),
        thist: Vec::new(),
        nhist: 0,
        tround: ZERO,
    };

    /* attach ops and content structures to overall structure */
    let interp: ARKInterp = Rc::new(_generic_ARKInterp {
        content: RefCell::new(Box::new(content)),
        ops: RefCell::new(ops),
    });

    /* fill content */
    let nmax = {
        let uround = ark_mem.borrow().uround;
        let mut c = lagrange_content_mut(&interp);

        /* maximum/current history length */
        c.nmax = SUNMIN(degree + 1, ARK_INTERP_MAX_DEGREE + 1); /* respect maximum possible */
        c.nmaxalloc = 0;
        c.nhist = 0;

        /* initialize time/solution history arrays to NULL */
        c.thist = Vec::new();
        c.yhist = Vec::new();

        /* initial t roundoff value */
        c.tround = FUZZ_FACTOR * uround;

        c.nmax
    };

    /* update workspace sizes */
    {
        let mut m = ark_mem.borrow_mut();
        m.lrw += (nmax + 1) as i64;
        m.liw += (nmax + 2) as i64;
    }

    Some(interp)
}

/*---------------------------------------------------------------
  arkInterpResize_Lagrange:

  This routine resizes the internal vectors.
  ---------------------------------------------------------------*/
pub fn arkInterpResize_Lagrange(
    ark_mem: &ARKodeMem,
    I: &ARKInterp,
    resize: Option<ARKVecResizeFn>,
    resize_data: &mut Option<Box<dyn Any>>,
    lrw_diff: sunindextype,
    liw_diff: sunindextype,
    y0: &N_Vector,
) -> i32 {
    /* resize vectors (the C `I == NULL` test is unrepresentable here) */
    let (have_yhist, nmaxalloc) = {
        let c = lagrange_content_mut(I);
        (!c.yhist.is_empty(), c.nmaxalloc)
    };
    if have_yhist {
        for i in 0..nmaxalloc as usize {
            let mut v = lagrange_content_mut(I).yhist[i].take();
            let ok = arkResizeVec(ark_mem, resize, resize_data, lrw_diff, liw_diff, y0, &mut v);
            lagrange_content_mut(I).yhist[i] = v;
            if !ok {
                return ARK_MEM_FAIL;
            }
        }
    }

    /* reset active history length */
    lagrange_content_mut(I).nhist = 0;

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  arkInterpFree_Lagrange:

  This routine frees the Lagrange ARKInterp structure.
  ---------------------------------------------------------------*/
pub fn arkInterpFree_Lagrange(ark_mem: &ARKodeMem, I: &ARKInterp) {
    /* if interpolation structure is NULL, just return -- unrepresentable */

    /* free content (C `I->content != NULL`) */
    let has_content = I.content.borrow().downcast_ref::<()>().is_none();
    if has_content {
        let (have_yhist, nmaxalloc) = {
            let c = lagrange_content_mut(I);
            (!c.yhist.is_empty(), c.nmaxalloc)
        };
        if have_yhist {
            for i in 0..nmaxalloc as usize {
                let mut v = lagrange_content_mut(I).yhist[i].take();
                if v.is_some() {
                    arkFreeVec(ark_mem, &mut v);
                }
                lagrange_content_mut(I).yhist[i] = v;
            }
            lagrange_content_mut(I).yhist = Vec::new();
        }
        if !lagrange_content_mut(I).thist.is_empty() {
            lagrange_content_mut(I).thist = Vec::new();
        }

        /* update work space sizes */
        {
            let nmax = lagrange_content_mut(I).nmax;
            let mut m = ark_mem.borrow_mut();
            m.lrw -= (nmax + 1) as i64;
            m.liw -= (nmax + 2) as i64;
        }

        /* C: free(I->content); I->content = NULL */
        *I.content.borrow_mut() = Box::new(());
    }

    /* free ops and interpolation structures: see arkInterpFree_Hermite */
    *I.ops.borrow_mut() = ARKInterpOps::default();
}

/*---------------------------------------------------------------
  arkInterpPrintMem_Lagrange

  This routine outputs the Lagrange temporal interpolation memory
  structure to a specified file pointer.
  ---------------------------------------------------------------*/
pub fn arkInterpPrintMem_Lagrange(I: &ARKInterp, outfile: &SUNFile) {
    /* the C `I != NULL` test is unrepresentable here */
    let c = lagrange_content_mut(I);
    outfile.write_str(&format!(
        "arkode_interp (Lagrange): nmax = {}\n",
        c.nmax
    ));
    outfile.write_str(&format!(
        "arkode_interp (Lagrange): nhist = {}\n",
        c.nhist
    ));
    if !c.thist.is_empty() {
        outfile.write_str("arkode_interp (Lagrange): thist =");
        for i in 0..c.nmax as usize {
            outfile.write_str(&format!("  {}", sun_format_g(c.thist[i])));
        }
        outfile.write_str("\n");
    }
    if !c.yhist.is_empty() {
        /* C prints the raw `N_Vector` pointers with "%p"; an `Rc` handle has
        no reproducible textual address, so a fixed placeholder is printed
        instead (deviation class 5 -- no reference example calls
        ARKodePrintMem). */
        outfile.write_str("arkode_interp (Lagrange): yhist ptrs =");
        for i in 0..c.nmax as usize {
            outfile.write_str(if c.yhist[i].is_some() {
                "  (vector)"
            } else {
                "  (nil)"
            });
        }
        outfile.write_str("\n");
    }
    /* SUNDIALS_DEBUG_PRINTVEC is not defined in the reference build */
}

/*---------------------------------------------------------------
  arkInterpSetDegree_Lagrange

  This routine sets a supplied interpolation degree which must be
  in the range 0 <= degree <= ARK_INTERP_MAX_DEGREE.

  Return values:
    ARK_ILL_INPUT -- if the input is outside of allowable bounds
    ARK_INTERP_FAIL -- if the interpolation module has already
       been initialized,
    ARK_SUCCESS -- successful completion.
  ---------------------------------------------------------------*/
pub fn arkInterpSetDegree_Lagrange(ark_mem: &ARKodeMem, I: &ARKInterp, degree: i32) -> i32 {
    if degree > ARK_INTERP_MAX_DEGREE || degree < 0 {
        arkProcessError(
            Some(ark_mem),
            ARK_INTERP_FAIL,
            line!() as i32,
            "arkInterpSetDegree_Lagrange",
            file!(),
            "Illegal degree specified.",
        );
        return ARK_ILL_INPUT;
    }

    lagrange_content_mut(I).nmax = degree + 1;

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  arkInterpInit_Lagrange

  This routine performs the following steps:
  1. allocates any missing/needed (t,y) history arrays
  2. zeros out stored (t,y) history
  3. copies current (t,y) from main ARKODE memory into history
  4. updates the 'active' history counter to 1
  ---------------------------------------------------------------*/
pub fn arkInterpInit_Lagrange(ark_mem: &ARKodeMem, I: &ARKInterp, tnew: sunrealtype) -> i32 {
    /* check if storage has increased since the last init */
    let (nmax, nmaxalloc) = {
        let c = lagrange_content_mut(I);
        (c.nmax, c.nmaxalloc)
    };
    if nmax > nmaxalloc {
        if !lagrange_content_mut(I).thist.is_empty() {
            lagrange_content_mut(I).thist = Vec::new();
        }
        if !lagrange_content_mut(I).yhist.is_empty() {
            for i in 0..nmaxalloc as usize {
                let mut v = lagrange_content_mut(I).yhist[i].take();
                if v.is_some() {
                    arkFreeVec(ark_mem, &mut v);
                }
                lagrange_content_mut(I).yhist[i] = v;
            }
            lagrange_content_mut(I).yhist = Vec::new();
        }
    }

    /* allocate storage for time and solution histories (C `malloc` failure
    branches are unreachable) */
    if lagrange_content_mut(I).thist.is_empty() {
        lagrange_content_mut(I).thist = vec![ZERO; nmax as usize];
    }

    /* solution history allocation */
    if lagrange_content_mut(I).yhist.is_empty() {
        lagrange_content_mut(I).yhist = vec![None; nmax as usize];
        for i in 0..nmax as usize {
            lagrange_content_mut(I).yhist[i] = None;
            let yn = ark_mem.borrow().yn.clone().expect("yn");
            let mut v = lagrange_content_mut(I).yhist[i].take();
            let ok = arkAllocVec(ark_mem, &yn, &mut v);
            lagrange_content_mut(I).yhist[i] = v;
            if !ok {
                arkInterpFree(ark_mem, Some(I));
                return ARK_MEM_FAIL;
            }
        }
    }

    /* update allocated size if necessary */
    {
        let mut c = lagrange_content_mut(I);
        if c.nmax > c.nmaxalloc {
            c.nmaxalloc = c.nmax;
        }
    }

    /* zero out history (to be safe) */
    let (nmaxalloc, yhist) = {
        let mut c = lagrange_content_mut(I);
        let nmaxalloc = c.nmaxalloc;
        for i in 0..nmaxalloc as usize {
            c.thist[i] = 0.0;
        }
        let yhist: Vec<N_Vector> = (0..nmaxalloc as usize)
            .map(|i| c.yhist[i].clone().expect("interp yhist"))
            .collect();
        (nmaxalloc, yhist)
    };
    if N_VConstVectorArray(nmaxalloc, 0.0, &yhist) != 0 {
        return ARK_VECTOROP_ERR;
    }

    /* set current time and state as first entries of (t,y) history, update counter */
    let y0 = {
        let mut c = lagrange_content_mut(I);
        c.thist[0] = tnew;
        c.yhist[0].clone().expect("interp yhist")
    };
    let yn = ark_mem.borrow().yn.clone().expect("yn");
    N_VScale(ONE, &yn, &y0);
    lagrange_content_mut(I).nhist = 1;

    /* return with success */
    ARK_SUCCESS
}

/*---------------------------------------------------------------
  arkInterpUpdate_Lagrange

  If the current time is 'different enough' from the stored
  values, then this routine performs the following steps:
  1. shifts the t-history array values, and prepends the current
     time
  2. shifts the y-history pointers, and copies the current state
     into the first history vector
  Otherwise it just returns with success.
  ---------------------------------------------------------------*/
pub fn arkInterpUpdate_Lagrange(ark_mem: &ARKodeMem, I: &ARKInterp, tnew: sunrealtype) -> i32 {
    /* set readability shortcuts */
    let (uround, tcur, h) = {
        let m = ark_mem.borrow();
        (m.uround, m.tcur, m.h)
    };

    let y0 = {
        let mut c = lagrange_content_mut(I);
        let nhist = c.nhist;
        let nmax = c.nmax;

        /* update t roundoff value */
        c.tround = FUZZ_FACTOR * uround * (SUNRabs(tcur) + SUNRabs(h));

        /* determine if tnew differs sufficiently from stored values */
        let mut tdiff = SUNRabs(tnew - c.thist[0]);
        for i in 1..nhist as usize {
            tdiff = SUNMIN(tdiff, SUNRabs(tnew - c.thist[i]));
        }
        if tdiff <= c.tround {
            return ARK_SUCCESS;
        }

        /* shift (t,y) history arrays by one */
        let ytmp = c.yhist[nmax as usize - 1].take();
        for i in (1..nmax as usize).rev() {
            c.thist[i] = c.thist[i - 1];
            let v = c.yhist[i - 1].take();
            c.yhist[i] = v;
        }
        c.yhist[0] = ytmp;

        /* copy tnew and ycur into first entry of history arrays */
        c.thist[0] = tnew;
        c.yhist[0].clone().expect("interp yhist")
    };
    let ycur = ark_mem.borrow().ycur.clone().expect("ycur");
    N_VScale(ONE, &ycur, &y0);

    /* update 'nhist' (first few steps) */
    {
        let mut c = lagrange_content_mut(I);
        let nhist = c.nhist;
        let nmax = c.nmax;
        c.nhist = SUNMIN(nhist + 1, nmax);
    }

    /* return with success */
    ARK_SUCCESS
}

/*---------------------------------------------------------------
  arkInterpEvaluate_Lagrange

  This routine evaluates a temporal interpolation/extrapolation
  based on the stored solution data in the interpolation structure.

  Derivatives have lower accuracy than the interpolant
  itself, losing one order per derivative.  This module can provide
  up to 3rd derivatives.

  The input 'tau' specifies the time at which to evaluate the
  Lagrange polynomial.  The formula for tau is defined using the
  most-recently-completed solution interval [t1,t0], and is
  given by:
               t = t0 + tau*(t0-t1),
  here t0 and t1 are the 2 most-recent entries in the 'thist'
  array within the interpolation structure.  Thus values
            -(nhist-1) <= tau < = 0
  provide interpolation, others result in extrapolation (assuming
  fixed step sizes, otherwise the stated lower bound is only
  approximate).
  ---------------------------------------------------------------*/
pub fn arkInterpEvaluate_Lagrange(
    ark_mem: &ARKodeMem,
    I: &ARKInterp,
    tau: sunrealtype,
    deriv: i32,
    degree: i32,
    yout: &N_Vector,
) -> i32 {
    /* local variables */
    let mut a: [sunrealtype; 6] = [ZERO; 6];

    /* set readability shortcuts */
    let nhist = lagrange_content_mut(I).nhist;

    /* determine polynomial degree q */
    let mut q = SUNMAX(degree, 0); /* respect lower bound */
    q = SUNMIN(q, nhist - 1); /* respect max possible */

    /* error on illegal deriv */
    if (deriv < 0) || (deriv > 3) {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "arkInterpEvaluate_Lagrange",
            file!(),
            "Requested illegal derivative.",
        );
        return ARK_ILL_INPUT;
    }

    /* if deriv is too high, just return zeros */
    if deriv > q {
        N_VConst(ZERO, yout);
        return ARK_SUCCESS;
    }

    /* if constant interpolant is requested, just return ynew */
    if q == 0 {
        let y0 = lagrange_content_mut(I).yhist[0]
            .clone()
            .expect("interp yhist");
        N_VScale(ONE, &y0, yout);
        return ARK_SUCCESS;
    }

    /* convert from tau back to t (both tnew and told are valid since q>0 => NHIST>1) */
    let tval = {
        let c = lagrange_content_mut(I);
        c.thist[0] + tau * (c.thist[0] - c.thist[1])
    };

    /* linear interpolant */
    if q == 1 {
        if deriv == 0 {
            a[0] = LBasis(I, 0, tval);
            a[1] = LBasis(I, 1, tval);
        } else {
            /* deriv == 1 */
            a[0] = LBasisD(I, 0, tval);
            a[1] = LBasisD(I, 1, tval);
        }
        let (y0, y1) = {
            let c = lagrange_content_mut(I);
            (
                c.yhist[0].clone().expect("interp yhist"),
                c.yhist[1].clone().expect("interp yhist"),
            )
        };
        N_VLinearSum(a[0], &y0, a[1], &y1, yout);
        return ARK_SUCCESS;
    }

    /* higher-degree interpolant */
    /*    initialize arguments for N_VLinearCombination */
    let X: Vec<N_Vector> = {
        let c = lagrange_content_mut(I);
        (0..(q + 1) as usize)
            .map(|i| c.yhist[i].clone().expect("interp yhist"))
            .collect()
    };
    for i in 0..(q + 1) as usize {
        a[i] = ZERO;
    }

    /*    construct linear combination coefficients based on derivative requested */
    match deriv {
        0 => {
            /* p(t) */
            for j in 0..(q + 1) {
                a[j as usize] = LBasis(I, j, tval);
            }
        }

        1 => {
            /* p'(t) */
            for j in 0..(q + 1) {
                a[j as usize] = LBasisD(I, j, tval);
            }
        }

        2 => {
            /* p''(t) */
            for j in 0..(q + 1) {
                a[j as usize] = LBasisD2(I, j, tval);
            }
        }

        3 => {
            /* p'''(t) */
            for j in 0..(q + 1) {
                a[j as usize] = LBasisD3(I, j, tval);
            }
        }

        _ => {}
    }

    /*    call N_VLinearCombination to evaluate the result, and return */
    let retval = N_VLinearCombination(q + 1, &a[..(q + 1) as usize], &X, yout);
    if retval != 0 {
        return ARK_VECTOROP_ERR;
    }

    ARK_SUCCESS
}

/* Lagrange utility routines (basis functions and their derivatives) */
pub fn LBasis(I: &ARKInterp, j: i32, t: sunrealtype) -> sunrealtype {
    let c = lagrange_content_mut(I);
    let mut p: sunrealtype = ONE;
    for k in 0..c.nhist {
        if k == j {
            continue;
        }
        p *= (t - c.thist[k as usize]) / (c.thist[j as usize] - c.thist[k as usize]);
    }
    p
}

pub fn LBasisD(I: &ARKInterp, j: i32, t: sunrealtype) -> sunrealtype {
    let c = lagrange_content_mut(I);
    let mut p: sunrealtype = ZERO;
    for i in 0..c.nhist {
        if i == j {
            continue;
        }
        let mut q: sunrealtype = ONE;
        for k in 0..c.nhist {
            if k == j {
                continue;
            }
            if k == i {
                continue;
            }
            q *= (t - c.thist[k as usize]) / (c.thist[j as usize] - c.thist[k as usize]);
        }
        p += q / (c.thist[j as usize] - c.thist[i as usize]);
    }

    p
}

pub fn LBasisD2(I: &ARKInterp, j: i32, t: sunrealtype) -> sunrealtype {
    let c = lagrange_content_mut(I);
    let mut p: sunrealtype = ZERO;
    for l in 0..c.nhist {
        if l == j {
            continue;
        }
        let mut q: sunrealtype = ZERO;
        for i in 0..c.nhist {
            if i == j {
                continue;
            }
            if i == l {
                continue;
            }
            let mut r: sunrealtype = ONE;
            for k in 0..c.nhist {
                if k == j {
                    continue;
                }
                if k == i {
                    continue;
                }
                if k == l {
                    continue;
                }
                r *= (t - c.thist[k as usize]) / (c.thist[j as usize] - c.thist[k as usize]);
            }
            q += r / (c.thist[j as usize] - c.thist[i as usize]);
        }
        p += q / (c.thist[j as usize] - c.thist[l as usize]);
    }

    p
}

pub fn LBasisD3(I: &ARKInterp, j: i32, t: sunrealtype) -> sunrealtype {
    let c = lagrange_content_mut(I);
    let mut p: sunrealtype = ZERO;
    for m in 0..c.nhist {
        if m == j {
            continue;
        }
        let mut q: sunrealtype = ZERO;
        for l in 0..c.nhist {
            if l == j {
                continue;
            }
            if l == m {
                continue;
            }
            let mut r: sunrealtype = ZERO;
            for i in 0..c.nhist {
                if i == j {
                    continue;
                }
                if i == m {
                    continue;
                }
                if i == l {
                    continue;
                }
                let mut s: sunrealtype = ONE;
                for k in 0..c.nhist {
                    if k == j {
                        continue;
                    }
                    if k == m {
                        continue;
                    }
                    if k == l {
                        continue;
                    }
                    if k == i {
                        continue;
                    }
                    s *= (t - c.thist[k as usize]) / (c.thist[j as usize] - c.thist[k as usize]);
                }
                r += s / (c.thist[j as usize] - c.thist[i as usize]);
            }
            q += r / (c.thist[j as usize] - c.thist[l as usize]);
        }
        p += q / (c.thist[j as usize] - c.thist[m as usize]);
    }

    p
}

/*===============================================================
  EOF
  ===============================================================*/
