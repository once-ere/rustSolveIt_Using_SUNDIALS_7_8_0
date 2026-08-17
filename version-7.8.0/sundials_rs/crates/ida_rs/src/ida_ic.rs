//! Port of `src/ida/ida_ic.c`: the consistent-initial-condition
//! calculation for IDA (`IDACalcIC`), with its own Newton iteration and
//! linesearch. It is independent of the linear solver in use.
//!
//! Fragment protocol: the module-scope `#define`s `ida_ic.c` repeats
//! (`ZERO`/`HALF`/`ONE`/`TWO`/`PT99`/`PT1`/`PT001`) and its dedicated IC
//! control constants (`ICRATEMAX`, `ALPHALS`, `IC_FAIL_RECOV`,
//! `IC_CONSTR_FAILED`, `IC_LINESRCH_FAILED`, `IC_CONV_FAIL`,
//! `IC_SLOW_CONVRG`) live in `ida_impl.rs` and are used from there
//! rather than redefined.
//!
//! Borrow discipline: every `IDA_mem` field access happens inside a
//! scoped block; no borrow is ever held across the residual callback,
//! the `ida_lsetup`/`ida_lsolve` calls, an `N_Vector` operation, the
//! error-weight function, `IDAWrmsNorm`, or `IDAProcessError` — all of
//! which reach the mem through their own handle.
//!
//! C `void*` callback data: `ida_user_data` / `ida_edata` are
//! `Option<Box<dyn Any>>` tokens that are `take`n for the duration of a
//! callback and restored on every path (`ida_edata == None` means "pass
//! the integrator `user_data`", matching `cvInitialSetup`'s
//! `cv_e_data = cv_user_data` pointer alias).

use crate::ida::{IDAInitialSetup, IDAWrmsNorm};
use crate::ida_impl::*;
use sundials_core::sundials_math::SUNRabs;
use sundials_core::sundials_nvector::{
    N_VClone, N_VConstrMask, N_VDestroy, N_VLinearSum, N_VMin, N_VMinQuotient, N_VProd, N_VScale,
};
use sundials_core::sundials_types::*;

/*
 * =================================================================
 * EXPORTED FUNCTIONS IMPLEMENTATION
 * =================================================================
 */

/*
 * -----------------------------------------------------------------
 * IDACalcIC
 * -----------------------------------------------------------------
 * IDACalcIC computes consistent initial conditions, given the
 * user's initial guess for unknown components of yy0 and/or yp0.
 *
 * The return value is IDA_SUCCESS = 0 if no error occurred.
 *
 * The error return values (fully described in ida.h) are:
 *   IDA_MEM_NULL        ida_mem is NULL
 *   IDA_NO_MALLOC       ida_mem was not allocated
 *   IDA_ILL_INPUT       bad value for icopt, tout1, or id
 *   IDA_LINIT_FAIL      the linear solver linit routine failed
 *   IDA_BAD_EWT         zero value of some component of ewt
 *   IDA_RES_FAIL        res had a non-recoverable error
 *   IDA_FIRST_RES_FAIL  res failed recoverably on the first call
 *   IDA_LSETUP_FAIL     lsetup had a non-recoverable error
 *   IDA_LSOLVE_FAIL     lsolve had a non-recoverable error
 *   IDA_NO_RECOVERY     res, lsetup, or lsolve had a recoverable
 *                       error, but IDACalcIC could not recover
 *   IDA_CONSTR_FAIL     the inequality constraints could not be met
 *   IDA_LINESEARCH_FAIL the linesearch failed (either on steptol test
 *                       or on the maxbacks test)
 *   IDA_CONV_FAIL       the Newton iterations failed to converge
 * -----------------------------------------------------------------
 */

pub fn IDACalcIC(ida_mem: &IDAMem, icopt: i32, tout1: sunrealtype) -> i32 {
    let ier: i32;
    let mxnh: i32;
    let mut retval: i32 = 0;
    let tdist: sunrealtype;
    let troundoff: sunrealtype;
    let mut hic: sunrealtype;
    let ypnorm: sunrealtype;

    /* Check if IDA memory exists: handled by the type system */
    let IDA_mem = ida_mem;

    /* Check if problem was malloc'ed */

    if IDA_mem.borrow().ida_MallocDone == SUNFALSE {
        IDAProcessError(
            Some(IDA_mem),
            IDA_NO_MALLOC,
            line!() as i32,
            "IDACalcIC",
            file!(),
            MSG_NO_MALLOC,
        );
        return IDA_NO_MALLOC;
    }

    /* Check inputs to IDA for correctness and consistency */

    ier = IDAInitialSetup(IDA_mem);
    if ier != IDA_SUCCESS {
        return IDA_ILL_INPUT;
    }
    IDA_mem.borrow_mut().ida_SetupDone = SUNTRUE;

    /* Check legality of input arguments, and set IDA memory copies. */

    if icopt != IDA_YA_YDP_INIT && icopt != IDA_Y_INIT {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDACalcIC",
            file!(),
            MSG_IC_BAD_ICOPT,
        );
        return IDA_ILL_INPUT;
    }
    IDA_mem.borrow_mut().ida_icopt = icopt;

    if icopt == IDA_YA_YDP_INIT && IDA_mem.borrow().ida_id.is_none() {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDACalcIC",
            file!(),
            MSG_IC_MISSING_ID,
        );
        return IDA_ILL_INPUT;
    }

    {
        let (tn, uround) = {
            let m = IDA_mem.borrow();
            (m.ida_tn, m.ida_uround)
        };
        tdist = SUNRabs(tout1 - tn);
        troundoff = TWO * uround * (SUNRabs(tn) + SUNRabs(tout1));
    }
    if tdist < troundoff {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDACalcIC",
            file!(),
            MSG_IC_TOO_CLOSE,
        );
        return IDA_ILL_INPUT;
    }

    /* Allocate space and initialize temporary vectors */

    let ee = { IDA_mem.borrow().ida_ee.clone() }.expect("ida_ee");
    /* C dereferences the clone results unconditionally (NULL => UB) */
    let yy0 = N_VClone(&ee).expect("N_VClone(ida_ee)");
    let yp0 = N_VClone(&ee).expect("N_VClone(ida_ee)");
    let (phi0, phi1) = {
        let mut m = IDA_mem.borrow_mut();
        m.ida_yy0 = Some(yy0.clone());
        m.ida_yp0 = Some(yp0.clone());
        m.ida_t0 = m.ida_tn;
        /* phi[0] and phi[1] handles are never reassigned inside IDACalcIC;
        hoisting the Rc clones is the locked move-state-into-locals pattern */
        (
            m.ida_phi[0].clone().expect("ida_phi[0]"),
            m.ida_phi[1].clone().expect("ida_phi[1]"),
        )
    };
    N_VScale(ONE, &phi0, &yy0);
    N_VScale(ONE, &phi1, &yp0);

    /* For use in the IDA_YA_YP_INIT case, set sysindex and tscale. */

    {
        let mut m = IDA_mem.borrow_mut();
        m.ida_sysindex = 1;
        m.ida_tscale = tdist;
    }
    if icopt == IDA_YA_YDP_INIT {
        let id = { IDA_mem.borrow().ida_id.clone() }.expect("ida_id");
        let minid = N_VMin(&id);
        if minid < ZERO {
            IDAProcessError(
                Some(IDA_mem),
                IDA_ILL_INPUT,
                line!() as i32,
                "IDACalcIC",
                file!(),
                MSG_IC_BAD_ID,
            );
            return IDA_ILL_INPUT;
        }
        if minid > HALF {
            IDA_mem.borrow_mut().ida_sysindex = 0;
        }
    }

    /* Set the test constant in the Newton convergence test */

    {
        let mut m = IDA_mem.borrow_mut();
        m.ida_epsNewt = m.ida_epiccon;
    }

    /* Initializations:
    cjratio = 1 (for use in direct linear solvers);
    set nbacktr = 0; */

    {
        let mut m = IDA_mem.borrow_mut();
        m.ida_cjratio = ONE;
        m.ida_nbacktr = 0;
    }

    /* Set hic, hh, cj, and mxnh. */

    hic = PT001 * tdist;
    let ewt = { IDA_mem.borrow().ida_ewt.clone() }.expect("ida_ewt");
    {
        let suppressalg = IDA_mem.borrow().ida_suppressalg;
        ypnorm = IDAWrmsNorm(IDA_mem, &yp0, &ewt, suppressalg);
    }
    if ypnorm > HALF / hic {
        hic = HALF / ypnorm;
    }
    if tout1 < IDA_mem.borrow().ida_tn {
        hic = -hic;
    }
    IDA_mem.borrow_mut().ida_hh = hic;
    if icopt == IDA_YA_YDP_INIT {
        let mut m = IDA_mem.borrow_mut();
        m.ida_cj = ONE / hic;
        mxnh = m.ida_maxnh;
    } else {
        IDA_mem.borrow_mut().ida_cj = ZERO;
        mxnh = 1;
    }

    /* Loop over nwt = number of evaluations of ewt vector. */

    for _nwt in 1..=2 {
        /* Loop over nh = number of h values. */
        for nh in 1..=mxnh {
            /* Call the IC nonlinear solver function. */
            retval = IDANlsIC(IDA_mem);

            /* Cut h and loop on recoverable IDA_YA_YDP_INIT failure; else break. */
            if retval == IDA_SUCCESS {
                break;
            }
            IDA_mem.borrow_mut().ida_ncfn += 1;
            if retval < 0 {
                break;
            }
            if nh == mxnh {
                break;
            }
            /* If looping to try again, reset yy0 and yp0 if not converging. */
            if retval != IC_SLOW_CONVRG {
                N_VScale(ONE, &phi0, &yy0);
                N_VScale(ONE, &phi1, &yp0);
            }
            hic *= PT1;
            {
                let mut m = IDA_mem.borrow_mut();
                m.ida_cj = ONE / hic;
                m.ida_hh = hic;
            }
        } /* End of nh loop */

        /* Break on failure; else reset ewt, save yy0, yp0 in phi, and loop. */
        if retval != IDA_SUCCESS {
            break;
        }
        let ewtsetOK: i32;
        {
            /* C: ewtsetOK = IDA_mem->ida_efun(yy0, ewt, IDA_mem->ida_edata).
            `ida_edata == None` marks the user-efun case where C aliases
            edata with user_data (boxes cannot alias). */
            let (efun, user_efun) = {
                let m = IDA_mem.borrow();
                (m.ida_efun, m.ida_user_efun)
            };
            let efun = efun.expect("ida_efun set");
            if user_efun {
                let mut data = IDA_mem.borrow_mut().ida_user_data.take();
                ewtsetOK = efun(&yy0, &ewt, &mut data);
                IDA_mem.borrow_mut().ida_user_data = data;
            } else {
                let mut data = IDA_mem.borrow_mut().ida_edata.take();
                ewtsetOK = efun(&yy0, &ewt, &mut data);
                IDA_mem.borrow_mut().ida_edata = data;
            }
        }
        if ewtsetOK != 0 {
            retval = IDA_BAD_EWT;
            break;
        }
        N_VScale(ONE, &yy0, &phi0);
        N_VScale(ONE, &yp0, &phi1);
    } /* End of nwt loop */

    /* Free temporary space */

    {
        let (yy0_owned, yp0_owned) = {
            let mut m = IDA_mem.borrow_mut();
            (m.ida_yy0.take(), m.ida_yp0.take())
        };
        drop(yy0);
        drop(yp0);
        if let Some(v) = yy0_owned {
            N_VDestroy(v);
        }
        if let Some(v) = yp0_owned {
            N_VDestroy(v);
        }
    }

    /* Load the optional outputs. */

    if icopt == IDA_YA_YDP_INIT {
        IDA_mem.borrow_mut().ida_hused = hic;
    }

    /* On any failure, print message and return proper flag. */

    if retval != IDA_SUCCESS {
        return IDAICFailFlag(IDA_mem, retval);
    }

    /* Otherwise return success flag. */

    IDA_SUCCESS
}

/*
 * =================================================================
 * PRIVATE FUNCTIONS IMPLEMENTATION
 * =================================================================
 */

/*
 * -----------------------------------------------------------------
 * IDANlsIC
 * -----------------------------------------------------------------
 * IDANlsIC solves a nonlinear system for consistent initial
 * conditions.  It calls IDANewtonIC to do most of the work.
 *
 * The return value is IDA_SUCCESS = 0 if no error occurred.
 * The error return values (positive) considered recoverable are:
 *  IC_FAIL_RECOV      if res, lsetup, or lsolve failed recoverably
 *  IC_CONSTR_FAILED   if the constraints could not be met
 *  IC_LINESRCH_FAILED if the linesearch failed (either on steptol test
 *                     or on maxbacks test)
 *  IC_CONV_FAIL       if the Newton iterations failed to converge
 *  IC_SLOW_CONVRG     if the iterations are converging slowly
 *                     (failed the convergence test, but showed
 *                     norm reduction or convergence rate < 1)
 * The error return values (negative) considered non-recoverable are:
 *  IDA_RES_FAIL       if res had a non-recoverable error
 *  IDA_FIRST_RES_FAIL if res failed recoverably on the first call
 *  IDA_LSETUP_FAIL    if lsetup had a non-recoverable error
 *  IDA_LSOLVE_FAIL    if lsolve had a non-recoverable error
 * -----------------------------------------------------------------
 */

fn IDANlsIC(IDA_mem: &IDAMem) -> i32 {
    let mut retval: i32;

    let (tv1, tv2, tv3, res, t0, yy0, yp0, delta, savres, maxnj) = {
        let m = IDA_mem.borrow();
        (
            m.ida_ee.clone().expect("ida_ee"),
            m.ida_tempv2.clone().expect("ida_tempv2"),
            m.ida_phi[2].clone().expect("ida_phi[2]"),
            m.ida_res.expect("ida_res"),
            m.ida_t0,
            m.ida_yy0.clone().expect("ida_yy0"),
            m.ida_yp0.clone().expect("ida_yp0"),
            m.ida_delta.clone().expect("ida_delta"),
            m.ida_savres.clone().expect("ida_savres"),
            m.ida_maxnj,
        )
    };

    /* Evaluate RHS. */
    let mut user_data = IDA_mem.borrow_mut().ida_user_data.take();
    retval = res(t0, &yy0, &yp0, &delta, &mut user_data);
    {
        let mut m = IDA_mem.borrow_mut();
        m.ida_user_data = user_data;
        m.ida_nre += 1;
    }
    if retval < 0 {
        return IDA_RES_FAIL;
    }
    if retval > 0 {
        return IDA_FIRST_RES_FAIL;
    }

    /* Save the residual. */
    N_VScale(ONE, &delta, &savres);

    /* Loop over nj = number of linear solve Jacobian setups. */

    for _nj in 1..=maxnj {
        /* If there is a setup routine, call it. */
        let lsetup = IDA_mem.borrow().ida_lsetup;
        if let Some(lsetup) = lsetup {
            IDA_mem.borrow_mut().ida_nsetups += 1;
            retval = lsetup(IDA_mem, &yy0, &yp0, &delta, &tv1, &tv2, &tv3);
            if retval < 0 {
                return IDA_LSETUP_FAIL;
            }
            if retval > 0 {
                return IC_FAIL_RECOV;
            }
        }

        /* Call the Newton iteration routine, and return if successful.  */
        retval = IDANewtonIC(IDA_mem);
        if retval == IDA_SUCCESS {
            return IDA_SUCCESS;
        }

        /* If converging slowly and lsetup is nontrivial, retry. */
        if retval == IC_SLOW_CONVRG && lsetup.is_some() {
            N_VScale(ONE, &savres, &delta);
            continue;
        } else {
            return retval;
        }
    } /* End of nj loop */

    /* No convergence after maxnj tries; return with retval=IC_SLOW_CONVRG */
    retval
}

/*
 * -----------------------------------------------------------------
 * IDANewtonIC
 * -----------------------------------------------------------------
 * IDANewtonIC performs the Newton iteration to solve for consistent
 * initial conditions.  It calls IDALineSrch within each iteration.
 * On return, savres contains the current residual vector.
 *
 * The return value is IDA_SUCCESS = 0 if no error occurred.
 * The error return values (positive) considered recoverable are:
 *  IC_FAIL_RECOV      if res or lsolve failed recoverably
 *  IC_CONSTR_FAILED   if the constraints could not be met
 *  IC_LINESRCH_FAILED if the linesearch failed (either on steptol test
 *                     or on maxbacks test)
 *  IC_CONV_FAIL       if the Newton iterations failed to converge
 *  IC_SLOW_CONVRG     if the iterations appear to be converging slowly.
 *                     They failed the convergence test, but showed
 *                     an overall norm reduction (by a factor of < 0.1)
 *                     or a convergence rate <= ICRATEMAX).
 * The error return values (negative) considered non-recoverable are:
 *  IDA_RES_FAIL   if res had a non-recoverable error
 *  IDA_LSOLVE_FAIL      if lsolve had a non-recoverable error
 * -----------------------------------------------------------------
 */

fn IDANewtonIC(IDA_mem: &IDAMem) -> i32 {
    let mut retval: i32;
    let mut fnorm: sunrealtype;
    let fnorm0: sunrealtype;
    let mut rate: sunrealtype;

    /* Set pointer for vector delnew */
    let phi2 = { IDA_mem.borrow().ida_phi[2].clone() }.expect("ida_phi[2]");
    IDA_mem.borrow_mut().ida_delnew = Some(phi2);

    /* Call the linear solve function to get the Newton step, delta. */
    let (lsolve, delta, ewt, yy0, yp0, savres) = {
        let m = IDA_mem.borrow();
        (
            m.ida_lsolve.expect("ida_lsolve"),
            m.ida_delta.clone().expect("ida_delta"),
            m.ida_ewt.clone().expect("ida_ewt"),
            m.ida_yy0.clone().expect("ida_yy0"),
            m.ida_yp0.clone().expect("ida_yp0"),
            m.ida_savres.clone().expect("ida_savres"),
        )
    };
    retval = lsolve(IDA_mem, &delta, &ewt, &yy0, &yp0, &savres);
    if retval < 0 {
        return IDA_LSOLVE_FAIL;
    }
    if retval > 0 {
        return IC_FAIL_RECOV;
    }

    /* Compute the norm of the step; return now if this is small. */
    fnorm = IDAWrmsNorm(IDA_mem, &delta, &ewt, SUNFALSE);
    {
        let m = IDA_mem.borrow();
        if m.ida_sysindex == 0 {
            fnorm *= m.ida_tscale * SUNRabs(m.ida_cj);
        }
    }
    if fnorm <= IDA_mem.borrow().ida_epsNewt {
        return IDA_SUCCESS;
    }
    fnorm0 = fnorm;

    /* Initialize rate to avoid compiler warning message */
    rate = ZERO;

    /* Newton iteration loop */

    let maxnit = IDA_mem.borrow().ida_maxnit;
    for _mnewt in 0..maxnit {
        IDA_mem.borrow_mut().ida_nni += 1;
        let mut delnorm = fnorm;
        let oldfnrm = fnorm;

        /* Call the Linesearch function and return if it failed. */
        retval = IDALineSrch(IDA_mem, &mut delnorm, &mut fnorm);
        if retval != IDA_SUCCESS {
            return retval;
        }

        /* Set the observed convergence rate and test for convergence. */
        rate = fnorm / oldfnrm;
        if fnorm <= IDA_mem.borrow().ida_epsNewt {
            return IDA_SUCCESS;
        }

        /* If not converged, copy new step vector, and loop. */
        let delnew = { IDA_mem.borrow().ida_delnew.clone() }.expect("ida_delnew");
        N_VScale(ONE, &delnew, &delta);
    } /* End of Newton iteration loop */

    /* Return either IC_SLOW_CONVRG or recoverable fail flag. */
    if rate <= ICRATEMAX || fnorm < PT1 * fnorm0 {
        return IC_SLOW_CONVRG;
    }
    IC_CONV_FAIL
}

/*
 * -----------------------------------------------------------------
 * IDALineSrch
 * -----------------------------------------------------------------
 * IDALineSrch performs the Linesearch algorithm with the
 * calculation of consistent initial conditions.
 *
 * On entry, yy0 and yp0 are the current values of y and y', the
 * Newton step is delta, the current residual vector F is savres,
 * delnorm is WRMS-norm(delta), and fnorm is the norm of the vector
 * J-inverse F.
 *
 * On a successful return, yy0, yp0, and savres have been updated,
 * delnew contains the current value of J-inverse F, and fnorm is
 * WRMS-norm(delnew).
 *
 * The return value is IDA_SUCCESS = 0 if no error occurred.
 * The error return values (positive) considered recoverable are:
 *  IC_FAIL_RECOV      if res or lsolve failed recoverably
 *  IC_CONSTR_FAILED   if the constraints could not be met
 *  IC_LINESRCH_FAILED if the linesearch failed (either on steptol test
 *                     or on maxbacks test)
 * The error return values (negative) considered non-recoverable are:
 *  IDA_RES_FAIL   if res had a non-recoverable error
 *  IDA_LSOLVE_FAIL      if lsolve had a non-recoverable error
 * -----------------------------------------------------------------
 */

fn IDALineSrch(IDA_mem: &IDAMem, delnorm: &mut sunrealtype, fnorm: &mut sunrealtype) -> i32 {
    let mut nbacks: i32;
    let f1norm: sunrealtype;
    let mut fnormp: sunrealtype = ZERO;
    let mut ratio: sunrealtype;
    let mut lambda: sunrealtype;
    let minlam: sunrealtype;
    let slpi: sunrealtype;

    /* Initialize work space pointers, f1norm, ratio.
    (Use of mc in constraint check does not conflict with ypnew.) */
    let (ee, tempv2, phi3) = {
        let m = IDA_mem.borrow();
        (
            m.ida_ee.clone().expect("ida_ee"),
            m.ida_tempv2.clone().expect("ida_tempv2"),
            m.ida_phi[3].clone().expect("ida_phi[3]"),
        )
    };
    let mc = ee.clone();
    {
        let mut m = IDA_mem.borrow_mut();
        m.ida_dtemp = Some(phi3);
        m.ida_ynew = Some(tempv2);
        m.ida_ypnew = Some(ee);
    }
    f1norm = (*fnorm) * (*fnorm) * HALF;
    ratio = ONE;

    /* If there are constraints, check and reduce step if necessary. */
    let constraints = { IDA_mem.borrow().ida_constraints.clone() };
    if let Some(constraints) = constraints {
        /* Update y and check constraints. */
        IDANewy(IDA_mem);
        let ynew = { IDA_mem.borrow().ida_ynew.clone() }.expect("ida_ynew");
        let conOK = N_VConstrMask(&constraints, &ynew, &mc);

        if !conOK {
            /* Not satisfied.  Compute scaled step to satisfy constraints. */
            let (delta, dtemp, yy0) = {
                let m = IDA_mem.borrow();
                (
                    m.ida_delta.clone().expect("ida_delta"),
                    m.ida_dtemp.clone().expect("ida_dtemp"),
                    m.ida_yy0.clone().expect("ida_yy0"),
                )
            };
            N_VProd(&mc, &delta, &dtemp);
            ratio = PT99 * N_VMinQuotient(&yy0, &dtemp);
            *delnorm *= ratio;
            if *delnorm <= IDA_mem.borrow().ida_steptol {
                return IC_CONSTR_FAILED;
            }
            N_VScale(ratio, &delta, &delta);
        }
    } /* End of constraints check */

    slpi = -TWO * f1norm * ratio;
    minlam = IDA_mem.borrow().ida_steptol / (*delnorm);
    lambda = ONE;
    nbacks = 0;

    /* In IDA_Y_INIT case, set ypnew = yp0 (fixed) for linesearch. */
    if IDA_mem.borrow().ida_icopt == IDA_Y_INIT {
        let (yp0, ypnew) = {
            let m = IDA_mem.borrow();
            (
                m.ida_yp0.clone().expect("ida_yp0"),
                m.ida_ypnew.clone().expect("ida_ypnew"),
            )
        };
        N_VScale(ONE, &yp0, &ypnew);
    }

    /* Loop on linesearch variable lambda. */

    loop {
        if nbacks == IDA_mem.borrow().ida_maxbacks {
            return IC_LINESRCH_FAILED;
        }
        /* Get new (y,y') = (ynew,ypnew) and norm of new function value. */
        IDANewyyp(IDA_mem, lambda);
        let retval = IDAfnorm(IDA_mem, &mut fnormp);
        if retval != IDA_SUCCESS {
            return retval;
        }

        /* If lsoff option is on, break out. */
        if IDA_mem.borrow().ida_lsoff {
            break;
        }

        /* Do alpha-condition test. */
        let f1normp = fnormp * fnormp * HALF;
        if f1normp <= f1norm + ALPHALS * slpi * lambda {
            break;
        }
        if lambda < minlam {
            return IC_LINESRCH_FAILED;
        }
        lambda /= TWO;
        IDA_mem.borrow_mut().ida_nbacktr += 1;
        nbacks += 1;
    } /* End of breakout linesearch loop */

    /* Update yy0, yp0, and fnorm, then return. */
    let (ynew, yy0) = {
        let m = IDA_mem.borrow();
        (
            m.ida_ynew.clone().expect("ida_ynew"),
            m.ida_yy0.clone().expect("ida_yy0"),
        )
    };
    N_VScale(ONE, &ynew, &yy0);
    if IDA_mem.borrow().ida_icopt == IDA_YA_YDP_INIT {
        let (ypnew, yp0) = {
            let m = IDA_mem.borrow();
            (
                m.ida_ypnew.clone().expect("ida_ypnew"),
                m.ida_yp0.clone().expect("ida_yp0"),
            )
        };
        N_VScale(ONE, &ypnew, &yp0);
    }
    *fnorm = fnormp;
    IDA_SUCCESS
}

/*
 * -----------------------------------------------------------------
 * IDAfnorm
 * -----------------------------------------------------------------
 * IDAfnorm computes the norm of the current function value, by
 * evaluating the DAE residual function, calling the linear
 * system solver, and computing a WRMS-norm.
 *
 * On return, savres contains the current residual vector F, and
 * delnew contains J-inverse F.
 *
 * The return value is IDA_SUCCESS = 0 if no error occurred, or
 *  IC_FAIL_RECOV    if res or lsolve failed recoverably, or
 *  IDA_RES_FAIL     if res had a non-recoverable error, or
 *  IDA_LSOLVE_FAIL  if lsolve had a non-recoverable error.
 * -----------------------------------------------------------------
 */

fn IDAfnorm(IDA_mem: &IDAMem, fnorm: &mut sunrealtype) -> i32 {
    let mut retval: i32;

    /* Get residual vector F, return if failed, and save F in savres. */
    let (res, t0, ynew, ypnew, delnew) = {
        let m = IDA_mem.borrow();
        (
            m.ida_res.expect("ida_res"),
            m.ida_t0,
            m.ida_ynew.clone().expect("ida_ynew"),
            m.ida_ypnew.clone().expect("ida_ypnew"),
            m.ida_delnew.clone().expect("ida_delnew"),
        )
    };
    let mut user_data = IDA_mem.borrow_mut().ida_user_data.take();
    retval = res(t0, &ynew, &ypnew, &delnew, &mut user_data);
    {
        let mut m = IDA_mem.borrow_mut();
        m.ida_user_data = user_data;
        m.ida_nre += 1;
    }
    if retval < 0 {
        return IDA_RES_FAIL;
    }
    if retval > 0 {
        return IC_FAIL_RECOV;
    }

    let savres = { IDA_mem.borrow().ida_savres.clone() }.expect("ida_savres");
    N_VScale(ONE, &delnew, &savres);

    /* Call the linear solve function to get J-inverse F; return if failed. */
    let (lsolve, ewt) = {
        let m = IDA_mem.borrow();
        (
            m.ida_lsolve.expect("ida_lsolve"),
            m.ida_ewt.clone().expect("ida_ewt"),
        )
    };
    retval = lsolve(IDA_mem, &delnew, &ewt, &ynew, &ypnew, &savres);
    if retval < 0 {
        return IDA_LSOLVE_FAIL;
    }
    if retval > 0 {
        return IC_FAIL_RECOV;
    }

    /* Compute the WRMS-norm; rescale if index = 0. */
    *fnorm = IDAWrmsNorm(IDA_mem, &delnew, &ewt, SUNFALSE);
    {
        let m = IDA_mem.borrow();
        if m.ida_sysindex == 0 {
            *fnorm *= m.ida_tscale * SUNRabs(m.ida_cj);
        }
    }

    IDA_SUCCESS
}

/*
 * -----------------------------------------------------------------
 * IDANewyyp
 * -----------------------------------------------------------------
 * IDANewyyp updates the vectors ynew and ypnew from yy0 and yp0,
 * using the current step vector lambda*delta, in a manner
 * depending on icopt and the input id vector.
 *
 * The return value is always IDA_SUCCESS = 0.
 * -----------------------------------------------------------------
 */

fn IDANewyyp(IDA_mem: &IDAMem, lambda: sunrealtype) -> i32 {
    /* IDA_YA_YDP_INIT case: ynew  = yy0 - lambda*delta    where id_i = 0
    ypnew = yp0 - cj*lambda*delta where id_i = 1. */
    if IDA_mem.borrow().ida_icopt == IDA_YA_YDP_INIT {
        let (id, delta, dtemp, yp0, cj, ypnew, yy0, ynew) = {
            let m = IDA_mem.borrow();
            (
                m.ida_id.clone().expect("ida_id"),
                m.ida_delta.clone().expect("ida_delta"),
                m.ida_dtemp.clone().expect("ida_dtemp"),
                m.ida_yp0.clone().expect("ida_yp0"),
                m.ida_cj,
                m.ida_ypnew.clone().expect("ida_ypnew"),
                m.ida_yy0.clone().expect("ida_yy0"),
                m.ida_ynew.clone().expect("ida_ynew"),
            )
        };
        N_VProd(&id, &delta, &dtemp);
        N_VLinearSum(ONE, &yp0, -cj * lambda, &dtemp, &ypnew);
        N_VLinearSum(ONE, &delta, -ONE, &dtemp, &dtemp);
        N_VLinearSum(ONE, &yy0, -lambda, &dtemp, &ynew);
        return IDA_SUCCESS;
    }

    /* IDA_Y_INIT case: ynew = yy0 - lambda*delta. (ypnew = yp0 preset.) */
    let (yy0, delta, ynew) = {
        let m = IDA_mem.borrow();
        (
            m.ida_yy0.clone().expect("ida_yy0"),
            m.ida_delta.clone().expect("ida_delta"),
            m.ida_ynew.clone().expect("ida_ynew"),
        )
    };
    N_VLinearSum(ONE, &yy0, -lambda, &delta, &ynew);
    IDA_SUCCESS
}

/*
 * -----------------------------------------------------------------
 * IDANewy
 * -----------------------------------------------------------------
 * IDANewy updates the vector ynew from yy0,
 * using the current step vector delta, in a manner
 * depending on icopt and the input id vector.
 *
 * The return value is always IDA_SUCCESS = 0.
 * -----------------------------------------------------------------
 */

fn IDANewy(IDA_mem: &IDAMem) -> i32 {
    /* IDA_YA_YDP_INIT case: ynew = yy0 - delta    where id_i = 0. */
    if IDA_mem.borrow().ida_icopt == IDA_YA_YDP_INIT {
        let (id, delta, dtemp, yy0, ynew) = {
            let m = IDA_mem.borrow();
            (
                m.ida_id.clone().expect("ida_id"),
                m.ida_delta.clone().expect("ida_delta"),
                m.ida_dtemp.clone().expect("ida_dtemp"),
                m.ida_yy0.clone().expect("ida_yy0"),
                m.ida_ynew.clone().expect("ida_ynew"),
            )
        };
        N_VProd(&id, &delta, &dtemp);
        N_VLinearSum(ONE, &delta, -ONE, &dtemp, &dtemp);
        N_VLinearSum(ONE, &yy0, -ONE, &dtemp, &ynew);
        return IDA_SUCCESS;
    }

    /* IDA_Y_INIT case: ynew = yy0 - delta. */
    let (yy0, delta, ynew) = {
        let m = IDA_mem.borrow();
        (
            m.ida_yy0.clone().expect("ida_yy0"),
            m.ida_delta.clone().expect("ida_delta"),
            m.ida_ynew.clone().expect("ida_ynew"),
        )
    };
    N_VLinearSum(ONE, &yy0, -ONE, &delta, &ynew);
    IDA_SUCCESS
}

/*
 * -----------------------------------------------------------------
 * IDAICFailFlag
 * -----------------------------------------------------------------
 * IDAICFailFlag prints a message and sets the IDACalcIC return
 * value appropriate to the flag retval returned by IDANlsIC.
 * -----------------------------------------------------------------
 */

fn IDAICFailFlag(IDA_mem: &IDAMem, retval: i32) -> i32 {
    /* Depending on retval, print error message and return error flag. */
    match retval {
        IDA_RES_FAIL => {
            IDAProcessError(
                Some(IDA_mem),
                IDA_RES_FAIL,
                line!() as i32,
                "IDAICFailFlag",
                file!(),
                MSG_IC_RES_NONREC,
            );
            IDA_RES_FAIL
        }

        IDA_FIRST_RES_FAIL => {
            IDAProcessError(
                Some(IDA_mem),
                IDA_FIRST_RES_FAIL,
                line!() as i32,
                "IDAICFailFlag",
                file!(),
                MSG_IC_RES_FAIL,
            );
            IDA_FIRST_RES_FAIL
        }

        IDA_LSETUP_FAIL => {
            IDAProcessError(
                Some(IDA_mem),
                IDA_LSETUP_FAIL,
                line!() as i32,
                "IDAICFailFlag",
                file!(),
                MSG_IC_SETUP_FAIL,
            );
            IDA_LSETUP_FAIL
        }

        IDA_LSOLVE_FAIL => {
            IDAProcessError(
                Some(IDA_mem),
                IDA_LSOLVE_FAIL,
                line!() as i32,
                "IDAICFailFlag",
                file!(),
                MSG_IC_SOLVE_FAIL,
            );
            IDA_LSOLVE_FAIL
        }

        IC_FAIL_RECOV => {
            IDAProcessError(
                Some(IDA_mem),
                IDA_NO_RECOVERY,
                line!() as i32,
                "IDAICFailFlag",
                file!(),
                MSG_IC_NO_RECOVERY,
            );
            IDA_NO_RECOVERY
        }

        IC_CONSTR_FAILED => {
            IDAProcessError(
                Some(IDA_mem),
                IDA_CONSTR_FAIL,
                line!() as i32,
                "IDAICFailFlag",
                file!(),
                MSG_IC_FAIL_CONSTR,
            );
            IDA_CONSTR_FAIL
        }

        IC_LINESRCH_FAILED => {
            IDAProcessError(
                Some(IDA_mem),
                IDA_LINESEARCH_FAIL,
                line!() as i32,
                "IDAICFailFlag",
                file!(),
                MSG_IC_FAILED_LINS,
            );
            IDA_LINESEARCH_FAIL
        }

        IC_CONV_FAIL => {
            IDAProcessError(
                Some(IDA_mem),
                IDA_CONV_FAIL,
                line!() as i32,
                "IDAICFailFlag",
                file!(),
                MSG_IC_CONV_FAILED,
            );
            IDA_CONV_FAIL
        }

        IC_SLOW_CONVRG => {
            IDAProcessError(
                Some(IDA_mem),
                IDA_CONV_FAIL,
                line!() as i32,
                "IDAICFailFlag",
                file!(),
                MSG_IC_CONV_FAILED,
            );
            IDA_CONV_FAIL
        }

        IDA_BAD_EWT => {
            IDAProcessError(
                Some(IDA_mem),
                IDA_BAD_EWT,
                line!() as i32,
                "IDAICFailFlag",
                file!(),
                MSG_IC_BAD_EWT,
            );
            IDA_BAD_EWT
        }

        _ => -99,
    }
}
