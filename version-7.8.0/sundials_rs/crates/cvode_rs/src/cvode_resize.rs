//! Port of `src/cvode/cvode_resize.c` (+ headers folded).
//!
//! Build Nordsieck array from solution history (CVodeResizeHistory and its
//! static helpers). C `sunrealtype*` / `N_Vector*` user arrays map to
//! `&[sunrealtype]` / `&[N_Vector]`; C NULL-entry checks on those arrays map
//! to slice-length checks (a NULL entry is unrepresentable — the equivalent
//! misuse is a too-short slice) returning the same `CV_ILL_INPUT` flag.

use crate::cvode_impl::*;
use sundials_core::sundials_math::{SUNMAX, SUNMIN};
use sundials_core::sundials_nonlinearsolver::SUNNonlinSolFree;
use sundials_core::sundials_nvector::{
    N_VClone, N_VConst, N_VDestroy, N_VLinearSum, N_VScale, N_Vector,
};
use sundials_core::sundials_types::*;
use sundials_core::sunnonlinsol_newton::SUNNonlinSol_Newton;

const ZERO: sunrealtype = 0.0; /* real 0.0 */
const ONE: sunrealtype = 1.0; /* real 1.0 */

/* -----------------------------------------------------------------------------
 * Build Adams Nordsieck array from f(t,y) history and y(t) value
 * ---------------------------------------------------------------------------*/

fn cvBuildNordsieckArrayAdams(
    t: &[sunrealtype],
    y: &N_Vector,
    f: &[N_Vector],
    wrk: &[N_Vector],
    order: i32,
    hscale: sunrealtype,
    zn: &[N_Vector],
) -> i32 {
    /* Check for valid inputs (C also NULL-checks t, y, f, wrk, and zn;
     * slices and references are non-null by construction) */
    if order < 1 {
        return CV_ILL_INPUT;
    }

    for i in 0..order {
        /* C: if (!f[i]) / if (!wrk[i]) — NULL entries map to short slices */
        if (i as usize) >= f.len() {
            return CV_ILL_INPUT;
        }
        if (i as usize) >= wrk.len() {
            return CV_ILL_INPUT;
        }
    }

    /* Compute Nordsieck array */
    if order > 1 {
        /* Compute Newton polynomial coefficients interpolating f history */
        for i in 0..order {
            N_VScale(ONE, &f[i as usize], &wrk[i as usize]);
        }

        for i in 1..order {
            let mut j = order - 1;
            while j >= i {
                /* Divided difference */
                let delta_t = ONE / (t[(j - i) as usize] - t[j as usize]);
                N_VLinearSum(
                    delta_t,
                    &wrk[(j - 1) as usize],
                    -delta_t,
                    &wrk[j as usize],
                    &wrk[j as usize],
                );
                j -= 1;
            }
        }

        /* Compute derivatives of Newton polynomial of f history */
        N_VScale(ONE, &wrk[(order - 1) as usize], &zn[1]);
        for i in 2..=order {
            N_VConst(ZERO, &zn[i as usize]);
        }

        let mut i = order - 2;
        while i >= 0 {
            let mut j = order - 1;
            while j > 0 {
                N_VLinearSum(
                    t[0] - t[i as usize],
                    &zn[(j + 1) as usize],
                    j as sunrealtype,
                    &zn[j as usize],
                    &zn[(j + 1) as usize],
                );
                j -= 1;
            }
            N_VLinearSum(t[0] - t[i as usize], &zn[1], ONE, &wrk[i as usize], &zn[1]);
            i -= 1;
        }
    }

    /* Overwrite first two columns with input values */
    N_VScale(ONE, y, &zn[0]);
    N_VScale(ONE, &f[0], &zn[1]);

    /* Scale entries */
    let mut scale = ONE;
    for i in 1..=order {
        scale *= hscale / (i as sunrealtype);
        N_VScale(scale, &zn[i as usize], &zn[i as usize]);
    }

    CV_SUCCESS
}

/* -----------------------------------------------------------------------------
 * Build BDF Nordsieck array from y(t) history and f(t,y) value
 * ---------------------------------------------------------------------------*/

fn cvBuildNordsieckArrayBDF(
    t: &[sunrealtype],
    y: &[N_Vector],
    f: &N_Vector,
    wrk: &[N_Vector],
    order: i32,
    hscale: sunrealtype,
    zn: &[N_Vector],
) -> i32 {
    /* Check for valid inputs (C also NULL-checks t, y, f, wrk, and zn;
     * slices and references are non-null by construction) */
    if order < 1 {
        return CV_ILL_INPUT;
    }

    for i in 0..order {
        /* C: if (!y[i]) — NULL entries map to short slices */
        if (i as usize) >= y.len() {
            return CV_ILL_INPUT;
        }
    }

    for i in 0..order + 1 {
        /* C: if (!wrk[i]) — NULL entries map to short slices */
        if (i as usize) >= wrk.len() {
            return CV_ILL_INPUT;
        }
    }

    /* Compute Nordsieck array */
    if order > 1 {
        /* Setup extended array of times to incorporate derivative value */
        let mut t_ext = [ZERO; BDF_Q_MAX + 1];

        t_ext[0] = t[0];
        for i in 1..=order {
            t_ext[i as usize] = t[(i - 1) as usize];
        }

        /* Compute Hermite polynomial coefficients interpolating y history and f */
        N_VScale(ONE, &y[0], &wrk[0]);
        for i in 1..=order {
            N_VScale(ONE, &y[(i - 1) as usize], &wrk[i as usize]);
        }

        for i in 1..=order {
            let mut j = order;
            while j > i - 1 {
                if i == 1 && j == 1 {
                    /* Replace with actual derivative value */
                    N_VScale(ONE, f, &wrk[j as usize]);
                } else {
                    /* Divided difference */
                    let delta_t = ONE / (t_ext[(j - i) as usize] - t_ext[j as usize]);
                    N_VLinearSum(
                        delta_t,
                        &wrk[(j - 1) as usize],
                        -delta_t,
                        &wrk[j as usize],
                        &wrk[j as usize],
                    );
                }
                j -= 1;
            }
        }

        /* Compute derivatives of Hermite polynomial */
        N_VScale(ONE, &wrk[order as usize], &zn[0]);
        for i in 1..=order {
            N_VConst(ZERO, &zn[i as usize]);
        }

        let mut i = order - 1;
        while i >= 0 {
            let mut j = order;
            while j > 0 {
                N_VLinearSum(
                    t_ext[0] - t_ext[i as usize],
                    &zn[j as usize],
                    j as sunrealtype,
                    &zn[(j - 1) as usize],
                    &zn[j as usize],
                );
                j -= 1;
            }
            N_VLinearSum(t_ext[0] - t_ext[i as usize], &zn[0], ONE, &wrk[i as usize], &zn[0]);
            i -= 1;
        }
    }

    /* Overwrite first two columns with input values */
    N_VScale(ONE, &y[0], &zn[0]);
    N_VScale(ONE, f, &zn[1]);

    /* Scale entries */
    let mut scale = ONE;
    for i in 1..=order {
        scale *= hscale / (i as sunrealtype);
        N_VScale(scale, &zn[i as usize], &zn[i as usize]);
    }

    CV_SUCCESS
}

/* -----------------------------------------------------------------------------
 * Compute predicted new state (simplified cvPredict for k = 1 and j = q...1)
 * ---------------------------------------------------------------------------*/

fn cvPredictY(order: i32, zn: &[N_Vector], ypred: &N_Vector) -> i32 {
    N_VScale(ONE, &zn[0], ypred);
    for j in 1..=order {
        N_VLinearSum(ONE, &zn[j as usize], ONE, ypred, ypred);
    }

    CV_SUCCESS
}

/* -----------------------------------------------------------------------------
 * Resize CVODE and build new history array
 * ---------------------------------------------------------------------------*/

pub fn CVodeResizeHistory(
    cvode_mem: &CVodeMem,
    t_hist: &[sunrealtype],
    y_hist: &[N_Vector],
    f_hist: &[N_Vector],
    num_y_hist: i32,
    num_f_hist: i32,
) -> i32 {
    /* ------------ *
     * Check inputs *
     * ------------ */

    /* NULL-mem check: handled by type system */

    /* C NULL checks on t_hist / y_hist / f_hist ("Time history array is
     * NULL" / "State history array is NULL" / "RHS history array is NULL"):
     * unrepresentable — slice references are non-null by construction. */

    /* Check that the input history is sufficient for the current (next) order */
    let (cv_q, cv_qmax, cv_lmm) = {
        let mem = cvode_mem.borrow();
        (mem.cv_q, mem.cv_qmax, mem.cv_lmm)
    };
    let n_hist: i32 = SUNMIN(cv_q + 1, cv_qmax);

    if cv_lmm == CV_ADAMS {
        if num_y_hist < 2 {
            cvProcessError(
                Some(cvode_mem),
                CV_ILL_INPUT,
                line!() as i32,
                "CVodeResizeHistory",
                file!(),
                "Insufficient solution history",
            );
            return CV_ILL_INPUT;
        }

        for i in 0..n_hist {
            /* C: if (!f_hist[i]) — NULL entries map to short slices */
            if (i as usize) >= f_hist.len() {
                cvProcessError(
                    Some(cvode_mem),
                    CV_ILL_INPUT,
                    line!() as i32,
                    "CVodeResizeHistory",
                    file!(),
                    "Insufficient right-hand side history",
                );
                return CV_ILL_INPUT;
            }
        }
    } else {
        if num_f_hist < 2 {
            cvProcessError(
                Some(cvode_mem),
                CV_ILL_INPUT,
                line!() as i32,
                "CVodeResizeHistory",
                file!(),
                "Insufficient right-hand side history",
            );
            return CV_ILL_INPUT;
        }

        for i in 0..n_hist {
            /* C: if (!y_hist[i]) — NULL entries map to short slices */
            if (i as usize) >= y_hist.len() {
                cvProcessError(
                    Some(cvode_mem),
                    CV_ILL_INPUT,
                    line!() as i32,
                    "CVodeResizeHistory",
                    file!(),
                    "Insufficient solution history",
                );
                return CV_ILL_INPUT;
            }
        }
    }

    /* -------------- *
     * Resize vectors *
     * -------------- */

    let old = cvode_mem.borrow_mut().cv_ewt.take();
    if let Some(v) = old {
        N_VDestroy(v);
    }
    match N_VClone(&y_hist[0]) {
        Some(v) => cvode_mem.borrow_mut().cv_ewt = Some(v),
        None => {
            cvProcessError(
                Some(cvode_mem),
                CV_MEM_FAIL,
                line!() as i32,
                "CVodeResizeHistory",
                file!(),
                "A vector allocation failed",
            );
            return CV_MEM_FAIL;
        }
    }

    let old = cvode_mem.borrow_mut().cv_acor.take();
    if let Some(v) = old {
        N_VDestroy(v);
    }
    match N_VClone(&y_hist[0]) {
        Some(v) => cvode_mem.borrow_mut().cv_acor = Some(v),
        None => {
            cvProcessError(
                Some(cvode_mem),
                CV_MEM_FAIL,
                line!() as i32,
                "CVodeResizeHistory",
                file!(),
                "A vector allocation failed",
            );
            return CV_MEM_FAIL;
        }
    }

    let old = cvode_mem.borrow_mut().cv_tempv.take();
    if let Some(v) = old {
        N_VDestroy(v);
    }
    match N_VClone(&y_hist[0]) {
        Some(v) => cvode_mem.borrow_mut().cv_tempv = Some(v),
        None => {
            cvProcessError(
                Some(cvode_mem),
                CV_MEM_FAIL,
                line!() as i32,
                "CVodeResizeHistory",
                file!(),
                "A vector allocation failed",
            );
            return CV_MEM_FAIL;
        }
    }

    let old = cvode_mem.borrow_mut().cv_ftemp.take();
    if let Some(v) = old {
        N_VDestroy(v);
    }
    match N_VClone(&y_hist[0]) {
        Some(v) => cvode_mem.borrow_mut().cv_ftemp = Some(v),
        None => {
            cvProcessError(
                Some(cvode_mem),
                CV_MEM_FAIL,
                line!() as i32,
                "CVodeResizeHistory",
                file!(),
                "A vector allocation failed",
            );
            return CV_MEM_FAIL;
        }
    }

    let old = cvode_mem.borrow_mut().cv_vtemp1.take();
    if let Some(v) = old {
        N_VDestroy(v);
    }
    match N_VClone(&y_hist[0]) {
        Some(v) => cvode_mem.borrow_mut().cv_vtemp1 = Some(v),
        None => {
            cvProcessError(
                Some(cvode_mem),
                CV_MEM_FAIL,
                line!() as i32,
                "CVodeResizeHistory",
                file!(),
                "A vector allocation failed",
            );
            return CV_MEM_FAIL;
        }
    }

    let old = cvode_mem.borrow_mut().cv_vtemp2.take();
    if let Some(v) = old {
        N_VDestroy(v);
    }
    match N_VClone(&y_hist[0]) {
        Some(v) => cvode_mem.borrow_mut().cv_vtemp2 = Some(v),
        None => {
            cvProcessError(
                Some(cvode_mem),
                CV_MEM_FAIL,
                line!() as i32,
                "CVodeResizeHistory",
                file!(),
                "A vector allocation failed",
            );
            return CV_MEM_FAIL;
        }
    }

    let old = cvode_mem.borrow_mut().cv_vtemp3.take();
    if let Some(v) = old {
        N_VDestroy(v);
    }
    match N_VClone(&y_hist[0]) {
        Some(v) => cvode_mem.borrow_mut().cv_vtemp3 = Some(v),
        None => {
            cvProcessError(
                Some(cvode_mem),
                CV_MEM_FAIL,
                line!() as i32,
                "CVodeResizeHistory",
                file!(),
                "A vector allocation failed",
            );
            return CV_MEM_FAIL;
        }
    }

    /* User will need to set a new vector of absolute tolerances */
    if cvode_mem.borrow().cv_VabstolMallocDone {
        let old = cvode_mem.borrow_mut().cv_Vabstol.take();
        if let Some(v) = old {
            N_VDestroy(v);
        }
        let new_vabstol = N_VClone(&y_hist[0]); /* C does not NULL-check this clone */
        cvode_mem.borrow_mut().cv_Vabstol = new_vabstol;
    }

    /* User will need to set a new constraints vector */
    let constraints = cvode_mem.borrow_mut().cv_constraints.take();
    if let Some(v) = constraints {
        N_VDestroy(v);
    }

    let cv_qmax_alloc = cvode_mem.borrow().cv_qmax_alloc;
    for j in 0..=cv_qmax_alloc {
        let old = cvode_mem.borrow_mut().cv_zn[j as usize].take();
        if let Some(v) = old {
            N_VDestroy(v);
        }
        match N_VClone(&y_hist[0]) {
            Some(v) => cvode_mem.borrow_mut().cv_zn[j as usize] = Some(v),
            None => {
                cvProcessError(
                    Some(cvode_mem),
                    CV_MEM_FAIL,
                    line!() as i32,
                    "CVodeResizeHistory",
                    file!(),
                    "A vector allocation failed",
                );
                return CV_MEM_FAIL;
            }
        }
    }

    /* ----------------------- *
     * Resize nonlinear solver *
     * ----------------------- */

    let (have_nls, own_nls) = {
        let mem = cvode_mem.borrow();
        (mem.NLS.is_some(), mem.ownNLS)
    };
    if have_nls && own_nls {
        let nls = cvode_mem.borrow_mut().NLS.take();
        let retval = SUNNonlinSolFree(nls);
        if retval != 0 {
            cvProcessError(
                Some(cvode_mem),
                CV_MEM_FAIL,
                line!() as i32,
                "CVodeResizeHistory",
                file!(),
                "Destroying the Newton solver failed",
            );
            return CV_MEM_FAIL;
        }
        /* cv_mem->NLS = NULL was performed by the take() above */
        cvode_mem.borrow_mut().ownNLS = SUNFALSE;

        let sunctx = cvode_mem.borrow().cv_sunctx.clone();
        let NLS = match SUNNonlinSol_Newton(&y_hist[0], &sunctx) {
            Some(nls) => nls,
            None => {
                cvProcessError(
                    Some(cvode_mem),
                    CV_MEM_FAIL,
                    line!() as i32,
                    "CVodeResizeHistory",
                    file!(),
                    "Error creating the Newton solver",
                );
                return CV_MEM_FAIL;
            }
        };

        let retval = crate::cvode_nls::CVodeSetNonlinearSolver(cvode_mem, &NLS);
        if retval != 0 {
            let _ = SUNNonlinSolFree(Some(NLS));
            cvProcessError(
                Some(cvode_mem),
                CV_MEM_FAIL,
                line!() as i32,
                "CVodeResizeHistory",
                file!(),
                "Error attaching default Newton solver",
            );
            return CV_MEM_FAIL;
        }
        cvode_mem.borrow_mut().ownNLS = SUNTRUE;
    }

    /* ----------------------------- *
     * Create workspace for resizing *
     * ----------------------------- */

    let (cv_qprime, cv_hscale) = {
        let mem = cvode_mem.borrow();
        (mem.cv_qprime, mem.cv_hscale)
    };

    let mut wrk_space_size: i32 = SUNMAX(cv_q, cv_qprime);
    if cv_lmm == CV_BDF {
        wrk_space_size += 1;
    }

    /* C: N_Vector resize_wrk[L_MAX] */
    let mut resize_wrk: Vec<N_Vector> = Vec::new();
    for _j in 0..wrk_space_size {
        match N_VClone(&y_hist[0]) {
            Some(v) => resize_wrk.push(v),
            None => {
                for v in resize_wrk {
                    N_VDestroy(v);
                }
                cvProcessError(
                    Some(cvode_mem),
                    CV_MEM_FAIL,
                    line!() as i32,
                    "CVodeResizeHistory",
                    file!(),
                    "A vector allocation failed",
                );
                return CV_MEM_FAIL;
            }
        }
    }

    /* Local handle array aliasing cv_mem->cv_zn (Rc clone = C pointer copy;
     * the helpers write through these handles into the mem's zn columns) */
    let zn: Vec<N_Vector> = {
        let mem = cvode_mem.borrow();
        (0..=cv_qmax_alloc as usize)
            .map(|j| mem.cv_zn[j].as_ref().unwrap().clone())
            .collect()
    };

    /* ------------------------------------------------------------------------ *
     * Construct Nordsieck array at the old time but with the new size to
     * compute correction vector at the new state size.
     * ------------------------------------------------------------------------ */

    if cv_q < cv_qmax {
        /* Compute z_{n-1} with new history size */
        let retval = if cv_lmm == CV_ADAMS {
            cvBuildNordsieckArrayAdams(
                &t_hist[1..],
                &y_hist[1],
                &f_hist[1..],
                &resize_wrk,
                cv_q,
                cv_hscale,
                &zn,
            )
        } else {
            cvBuildNordsieckArrayBDF(
                &t_hist[1..],
                &y_hist[1..],
                &f_hist[1],
                &resize_wrk,
                cv_q,
                cv_hscale,
                &zn,
            )
        };

        if retval != 0 {
            cvProcessError(
                Some(cvode_mem),
                retval,
                line!() as i32,
                "CVodeResizeHistory",
                file!(),
                "Building the Nordsieck array failed",
            );
            return retval;
        }

        /* Get predicted value */
        let vtemp1 = cvode_mem.borrow().cv_vtemp1.as_ref().unwrap().clone();
        let retval = cvPredictY(cv_q, &zn, &vtemp1);

        if retval != 0 {
            cvProcessError(
                Some(cvode_mem),
                retval,
                line!() as i32,
                "CVodeResizeHistory",
                file!(),
                "Computing the predictor failed",
            );
            return retval;
        }

        /* Resized correction */
        N_VLinearSum(ONE, &y_hist[0], -ONE, &vtemp1, &zn[cv_qmax as usize]);
    }

    /* ----------------------------- *
     * Construct new Nordsieck Array *
     * ----------------------------- */

    let retval = if cv_lmm == CV_ADAMS {
        cvBuildNordsieckArrayAdams(
            t_hist,
            &y_hist[0],
            f_hist,
            &resize_wrk,
            cv_qprime,
            cv_hscale,
            &zn,
        )
    } else {
        cvBuildNordsieckArrayBDF(
            t_hist,
            y_hist,
            &f_hist[0],
            &resize_wrk,
            cv_qprime,
            cv_hscale,
            &zn,
        )
    };

    if retval != 0 {
        cvProcessError(
            Some(cvode_mem),
            retval,
            line!() as i32,
            "CVodeResizeHistory",
            file!(),
            "Building the Nordsieck array failed",
        );
        return retval;
    }

    /* ------------------- *
     * Update time history *
     * ------------------- */

    /* Ensure internal time and step history match the input history */
    {
        let mut mem = cvode_mem.borrow_mut();
        mem.cv_tn = t_hist[0];

        for i in 1..n_hist {
            mem.cv_tau[i as usize] = t_hist[(i - 1) as usize] - t_hist[i as usize];
        }

        /* In the next step, perform initialization needed after a resize */
        mem.first_step_after_resize = SUNTRUE;
    }

    /* ------------------------------ *
     * Destroy workspace for resizing *
     * ------------------------------ */

    for v in resize_wrk {
        N_VDestroy(v);
    }

    CV_SUCCESS
}
