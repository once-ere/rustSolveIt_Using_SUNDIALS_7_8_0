//! Port of `src/idas/idas_ic.c`: the consistent-initial-condition
//! calculation for IDAS (`IDACalcIC`), with its own Newton iteration and
//! linesearch, plus the sensitivity variants (`IDASensNlsIC`,
//! `IDASensNewtonIC`, `IDASensLineSrch`, `IDASensfnorm`,
//! `IDASensNewyyp`). It is independent of the linear solver in use.
//!
//! IDAS extends IDA's IC calculation two ways:
//!   * `IDA_SIMULTANEOUS` sensitivities ride along inside the state
//!     Newton/linesearch (the `sensi_sim` branches of `IDANlsIC`,
//!     `IDANewtonIC`, `IDALineSrch`, `IDAfnorm`, `IDANewyyp`);
//!   * `IDA_STAGGERED` sensitivities get their own second
//!     nwt/nh/Newton/linesearch pass after the state IC has converged.
//!
//! Fragment protocol: the module-scope `#define`s `idas_ic.c` repeats
//! (`ZERO`/`HALF`/`ONE`/`TWO`/`PT99`/`PT1`/`PT001`) and its dedicated IC
//! control constants (`ICRATEMAX`, `ALPHALS`, `IC_FAIL_RECOV`,
//! `IC_CONSTR_FAILED`, `IC_LINESRCH_FAILED`, `IC_CONV_FAIL`,
//! `IC_SLOW_CONVRG`) live in `idas_impl.rs` and are used from there
//! rather than redefined.
//!
//! Borrow discipline: every `IDA_mem` field access happens inside a
//! scoped block; no borrow is ever held across the residual callback,
//! the sensitivity residual callback, the `ida_lsetup`/`ida_lsolve`
//! calls, an `N_Vector` operation, the error-weight functions,
//! `IDAWrmsNorm`/`IDASensWrmsNorm[Update]`, or `IDAProcessError` — all of
//! which reach the mem through their own handle.
//!
//! C `void*` callback data: `ida_user_data` / `ida_edata` /
//! `ida_user_dataS` are `Option<Box<dyn Any>>` tokens that are `take`n
//! for the duration of a callback and restored on every path
//! (`ida_edata == None` means "pass the integrator `user_data`", matching
//! C's `ida_edata = ida_user_data` pointer alias set in
//! `IDAInitialSetup`; `ida_user_dataS` holds an `IDAMem` clone when the
//! internal DQ sensitivity residual is in use, matching C's
//! `ida_user_dataS = (void*)IDA_mem`).
//!
//! C `N_Vector*` sensitivity arrays are `Vec<N_Vector>` of handle clones,
//! so the C pointer-array assignments
//! (`ida_savresS = ida_phiS[2]`, `ida_delnewS = ida_phiS[3]`,
//! `ida_yyS0new = ida_phiS[4]`, `ida_ypS0new = ida_eeS`) become `Vec`
//! clones that alias exactly the same vectors (`Rc` clone == C pointer
//! copy).

use crate::idas::{
    IDAInitialSetup, IDASensEwtSet, IDASensWrmsNorm, IDASensWrmsNormUpdate, IDAWrmsNorm,
};
use crate::idas_impl::*;
use sundials_core::sundials_math::SUNRabs;
use sundials_core::sundials_nvector::{
    N_VClone, N_VCloneVectorArray, N_VConstrMask, N_VDestroy, N_VDestroyVectorArray, N_VLinearSum,
    N_VMin, N_VMinQuotient, N_VProd, N_VScale, N_Vector,
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
    let mut ypnorm: sunrealtype;
    let sensi_stg: sunbooleantype;
    let sensi_sim: sunbooleantype;

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

    /* Are we computing sensitivities? */
    let (sensi, Ns) = {
        let m = IDA_mem.borrow();
        sensi_stg = m.ida_sensi && (m.ida_ism == IDA_STAGGERED);
        sensi_sim = m.ida_sensi && (m.ida_ism == IDA_SIMULTANEOUS);
        (m.ida_sensi, m.ida_Ns)
    };

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

    /* phiS[0], phiS[1] are likewise fixed for the whole call; empty when
    sensitivities are off (never indexed on that path). */
    let mut phiS0: Vec<N_Vector> = Vec::new();
    let mut phiS1: Vec<N_Vector> = Vec::new();
    let mut yyS0: Vec<N_Vector> = Vec::new();
    let mut ypS0: Vec<N_Vector> = Vec::new();

    if sensi {
        /* Allocate temporary space required for sensitivity IC: yyS0 and ypS0. */
        yyS0 = N_VCloneVectorArray(Ns, &ee).expect("N_VCloneVectorArray(ida_ee)");
        ypS0 = N_VCloneVectorArray(Ns, &ee).expect("N_VCloneVectorArray(ida_ee)");
        {
            let mut m = IDA_mem.borrow_mut();
            m.ida_yyS0 = yyS0.clone();
            m.ida_ypS0 = ypS0.clone();
            phiS0 = m.ida_phiS[0].clone();
            phiS1 = m.ida_phiS[1].clone();
        }

        /* Initialize sensitivity vector. */
        for is in 0..(Ns as usize) {
            N_VScale(ONE, &phiS0[is], &yyS0[is]);
            N_VScale(ONE, &phiS1[is], &ypS0[is]);
        }

        /* Initialize work space vectors needed for sensitivities.
        (C pointer-array assignments; `Vec` clones alias the same vectors.) */
        let (phiS2, phiS3, phiS4, eeS) = {
            let m = IDA_mem.borrow();
            (
                m.ida_phiS[2].clone(),
                m.ida_phiS[3].clone(),
                m.ida_phiS[4].clone(),
                m.ida_eeS.clone(),
            )
        };
        {
            let mut m = IDA_mem.borrow_mut();
            m.ida_savresS = phiS2;
            m.ida_delnewS = phiS3;
            m.ida_yyS0new = phiS4;
            m.ida_ypS0new = eeS;
        }
    }

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

    if sensi_sim {
        let ewtS = { IDA_mem.borrow().ida_ewtS.clone() };
        ypnorm = IDASensWrmsNormUpdate(IDA_mem, ypnorm, &ypS0[..], &ewtS[..], SUNFALSE);
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
                if sensi_sim {
                    /* Reset yyS0 and ypS0. */
                    /* Copy phiS[0] and phiS[1] into yyS0 and ypS0. */
                    for is in 0..(Ns as usize) {
                        N_VScale(ONE, &phiS0[is], &yyS0[is]);
                        N_VScale(ONE, &phiS1[is], &ypS0[is]);
                    }
                }
            }
            hic *= PT1;
            {
                let mut m = IDA_mem.borrow_mut();
                m.ida_cj = ONE / hic;
                m.ida_hh = hic;
            }
        } /* End of nh loop */

        /* Break on failure */
        if retval != IDA_SUCCESS {
            break;
        }

        /* Reset ewt, save yy0, yp0 in phi, and loop. */
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

        if sensi_sim {
            /* Reevaluate ewtS. */
            let ewtS = { IDA_mem.borrow().ida_ewtS.clone() };
            let ewtsetOK = IDASensEwtSet(IDA_mem, &yyS0[..], &ewtS[..]);
            if ewtsetOK != 0 {
                retval = IDA_BAD_EWT;
                break;
            }

            /* Save yyS0 and ypS0. */
            for is in 0..(Ns as usize) {
                N_VScale(ONE, &yyS0[is], &phiS0[is]);
                N_VScale(ONE, &ypS0[is], &phiS1[is]);
            }
        }
    } /* End of nwt loop */

    /* Load the optional outputs. */

    if icopt == IDA_YA_YDP_INIT {
        IDA_mem.borrow_mut().ida_hused = hic;
    }

    /* On any failure, free memory, print error message and return */

    if retval != IDA_SUCCESS {
        idaicFreeTempSpace(IDA_mem, sensi, Ns, yy0, yp0, yyS0, ypS0);

        let icret = IDAICFailFlag(IDA_mem, retval);
        return icret;
    }

    /* Unless using the STAGGERED approach for sensitivities, return now */

    if !sensi_stg {
        idaicFreeTempSpace(IDA_mem, sensi, Ns, yy0, yp0, yyS0, ypS0);

        return IDA_SUCCESS;
    }

    /* Find consistent I.C. for sensitivities using a staggered approach */

    /* Evaluate res at converged y, needed for future evaluations of sens. RHS
    If res() fails recoverably, treat it as a convergence failure and
    attempt the step again */

    {
        let (res, t0, delta) = {
            let m = IDA_mem.borrow();
            (
                m.ida_res.expect("ida_res"),
                m.ida_t0,
                m.ida_delta.clone().expect("ida_delta"),
            )
        };
        let mut user_data = IDA_mem.borrow_mut().ida_user_data.take();
        retval = res(t0, &yy0, &yp0, &delta, &mut user_data);
        {
            let mut m = IDA_mem.borrow_mut();
            m.ida_user_data = user_data;
            m.ida_nre += 1;
        }
    }
    if retval < 0 {
        /* res function failed unrecoverably. */
        return IDA_RES_FAIL;
    }

    if retval > 0 {
        /* res function failed recoverably but no recovery possible. */
        return IDA_FIRST_RES_FAIL;
    }

    /* Loop over nwt = number of evaluations of ewt vector. */
    for _nwt in 1..=2 {
        /* Loop over nh = number of h values. */
        for nh in 1..=mxnh {
            retval = IDASensNlsIC(IDA_mem);
            if retval == IDA_SUCCESS {
                break;
            }

            /* Increment the number of the sensitivity related corrector convergence failures. */
            IDA_mem.borrow_mut().ida_ncfnS += 1;

            if retval < 0 {
                break;
            }
            if nh == mxnh {
                break;
            }

            /* If looping to try again, reset yyS0 and ypS0 if not converging. */
            if retval != IC_SLOW_CONVRG {
                for is in 0..(Ns as usize) {
                    N_VScale(ONE, &phiS0[is], &yyS0[is]);
                    N_VScale(ONE, &phiS1[is], &ypS0[is]);
                }
            }
            hic *= PT1;
            {
                let mut m = IDA_mem.borrow_mut();
                m.ida_cj = ONE / hic;
                m.ida_hh = hic;
            }
        } /* End of nh loop */

        /* Break on failure */
        if retval != IDA_SUCCESS {
            break;
        }

        /* Since it was successful, reevaluate ewtS with the new values of yyS0, save
        yyS0 and ypS0 in phiS[0] and phiS[1] and loop one more time to check and
        maybe correct the  new sensitivities IC with respect to the new weights. */

        /* Reevaluate ewtS. */
        let ewtS = { IDA_mem.borrow().ida_ewtS.clone() };
        let ewtsetOK = IDASensEwtSet(IDA_mem, &yyS0[..], &ewtS[..]);
        if ewtsetOK != 0 {
            retval = IDA_BAD_EWT;
            break;
        }

        /* Save yyS0 and ypS0. */
        for is in 0..(Ns as usize) {
            N_VScale(ONE, &yyS0[is], &phiS0[is]);
            N_VScale(ONE, &ypS0[is], &phiS1[is]);
        }
    } /* End of nwt loop */

    /* Load the optional outputs. */
    if icopt == IDA_YA_YDP_INIT {
        IDA_mem.borrow_mut().ida_hused = hic;
    }

    /* Free temporary space */
    /* Here sensi is SUNTRUE, so deallocate sensitivity temporary vectors. */
    idaicFreeTempSpace(IDA_mem, SUNTRUE, Ns, yy0, yp0, yyS0, ypS0);

    /* On any failure, print message and return proper flag. */
    if retval != IDA_SUCCESS {
        let icret = IDAICFailFlag(IDA_mem, retval);
        return icret;
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
 * The C free block repeated verbatim at the three IDACalcIC exit points:
 *
 *   N_VDestroy(IDA_mem->ida_yy0);
 *   N_VDestroy(IDA_mem->ida_yp0);
 *   if (IDA_mem->ida_sensi) {
 *     N_VDestroyVectorArray(IDA_mem->ida_yyS0, IDA_mem->ida_Ns);
 *     N_VDestroyVectorArray(IDA_mem->ida_ypS0, IDA_mem->ida_Ns);
 *   }
 *
 * The handles are taken out of the mem and the caller's local clones are
 * moved in, so the destroy consumes the last owner (C leaves the mem
 * fields dangling here; nothing reads them afterwards).
 */
fn idaicFreeTempSpace(
    IDA_mem: &IDAMem,
    sensi: sunbooleantype,
    Ns: i32,
    yy0: N_Vector,
    yp0: N_Vector,
    yyS0: Vec<N_Vector>,
    ypS0: Vec<N_Vector>,
) {
    drop(yy0);
    drop(yp0);
    drop(yyS0);
    drop(ypS0);

    let (m_yy0, m_yp0) = {
        let mut m = IDA_mem.borrow_mut();
        (m.ida_yy0.take(), m.ida_yp0.take())
    };
    if let Some(v) = m_yy0 {
        N_VDestroy(v);
    }
    if let Some(v) = m_yp0 {
        N_VDestroy(v);
    }

    if sensi {
        let (m_yyS0, m_ypS0) = {
            let mut m = IDA_mem.borrow_mut();
            (
                std::mem::take(&mut m.ida_yyS0),
                std::mem::take(&mut m.ida_ypS0),
            )
        };
        N_VDestroyVectorArray(m_yyS0, Ns);
        N_VDestroyVectorArray(m_ypS0, Ns);
    }
}

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
    let sensi_sim: sunbooleantype;

    /* Are we computing sensitivities with the IDA_SIMULTANEOUS approach? */
    let (tv1, tv2, tv3, res, t0, yy0, yp0, delta, savres, maxnj, Ns) = {
        let m = IDA_mem.borrow();
        sensi_sim = m.ida_sensi && (m.ida_ism == IDA_SIMULTANEOUS);
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
            m.ida_Ns,
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

    /* deltaS/savresS handles are fixed for the whole call (deltaS is
    allocated by IDASensInit, savresS aliases phiS[2]); empty and unused
    unless sensi_sim. */
    let mut deltaS: Vec<N_Vector> = Vec::new();
    let mut savresS: Vec<N_Vector> = Vec::new();

    if sensi_sim {
        /*Evaluate sensitivity RHS and save it in savresS. */
        let (resS, yyS0, ypS0, tmpS1, tmpS2, tmpS3) = {
            let m = IDA_mem.borrow();
            deltaS = m.ida_deltaS.clone();
            savresS = m.ida_savresS.clone();
            (
                m.ida_resS.expect("ida_resS"),
                m.ida_yyS0.clone(),
                m.ida_ypS0.clone(),
                m.ida_tmpS1.clone().expect("ida_tmpS1"),
                m.ida_tmpS2.clone().expect("ida_tmpS2"),
                m.ida_tmpS3.clone().expect("ida_tmpS3"),
            )
        };
        let mut user_dataS = IDA_mem.borrow_mut().ida_user_dataS.take();
        /* C: `ida_user_dataS` is `IDA_mem` when the internal DQ residual is in
        use and `ida_user_data` otherwise (idas.c:1359/1365). Invariant D:
        `Some(box)` is the module-owned token, `None` means hand over
        `ida_user_data`. */
        let resS_from_user_data = user_dataS.is_none();
        if resS_from_user_data {
            user_dataS = IDA_mem.borrow_mut().ida_user_data.take();
        }
        retval = resS(
            Ns,
            t0,
            &yy0,
            &yp0,
            &delta,
            &yyS0[..],
            &ypS0[..],
            &deltaS[..],
            &mut user_dataS,
            &tmpS1,
            &tmpS2,
            &tmpS3,
        );
        {
            let mut m = IDA_mem.borrow_mut();
            if resS_from_user_data {
                m.ida_user_data = user_dataS;
            } else {
                m.ida_user_dataS = user_dataS;
            }
            m.ida_nrSe += 1;
        }
        if retval < 0 {
            return IDA_RES_FAIL;
        }
        if retval > 0 {
            return IDA_FIRST_RES_FAIL;
        }

        for is in 0..(Ns as usize) {
            N_VScale(ONE, &deltaS[is], &savresS[is]);
        }
    }

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

            if sensi_sim {
                for is in 0..(Ns as usize) {
                    N_VScale(ONE, &savresS[is], &deltaS[is]);
                }
            }

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
    let sensi_sim: sunbooleantype;

    /* Are we computing sensitivities with the IDA_SIMULTANEOUS approach? */
    let Ns = {
        let m = IDA_mem.borrow();
        sensi_sim = m.ida_sensi && (m.ida_ism == IDA_SIMULTANEOUS);
        m.ida_Ns
    };

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

    /* Compute the norm of the step. */
    fnorm = IDAWrmsNorm(IDA_mem, &delta, &ewt, SUNFALSE);

    /* Call the lsolve function to get correction vectors deltaS. */
    let mut deltaS: Vec<N_Vector> = Vec::new();
    if sensi_sim {
        let (deltaS_v, ewtS) = {
            let m = IDA_mem.borrow();
            (m.ida_deltaS.clone(), m.ida_ewtS.clone())
        };
        deltaS = deltaS_v;
        for is in 0..(Ns as usize) {
            retval = lsolve(IDA_mem, &deltaS[is], &ewtS[is], &yy0, &yp0, &savres);
            if retval < 0 {
                return IDA_LSOLVE_FAIL;
            }
            if retval > 0 {
                return IC_FAIL_RECOV;
            }
        }
        /* Update the norm of delta. */
        fnorm = IDASensWrmsNormUpdate(IDA_mem, fnorm, &deltaS[..], &ewtS[..], SUNFALSE);
    }

    /* Test for convergence. Return now if the norm is small. */
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

        if sensi_sim {
            /* Update the iteration's step for sensitivities. */
            let delnewS = { IDA_mem.borrow().ida_delnewS.clone() };
            for is in 0..(Ns as usize) {
                N_VScale(ONE, &delnewS[is], &deltaS[is]);
            }
        }
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
    let sensi_sim: sunbooleantype;

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

    /* Are we computing sensitivities with the IDA_SIMULTANEOUS approach? */
    let Ns = {
        let m = IDA_mem.borrow();
        sensi_sim = m.ida_sensi && (m.ida_ism == IDA_SIMULTANEOUS);
        m.ida_Ns
    };

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

        /* do the same for sensitivities. */
        if sensi_sim {
            let (ypS0, ypS0new) = {
                let m = IDA_mem.borrow();
                (m.ida_ypS0.clone(), m.ida_ypS0new.clone())
            };
            for is in 0..(Ns as usize) {
                N_VScale(ONE, &ypS0[is], &ypS0new[is]);
            }
        }
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

    /* Update yy0, yp0. */
    let (ynew, yy0) = {
        let m = IDA_mem.borrow();
        (
            m.ida_ynew.clone().expect("ida_ynew"),
            m.ida_yy0.clone().expect("ida_yy0"),
        )
    };
    N_VScale(ONE, &ynew, &yy0);

    if sensi_sim {
        /* Update yyS0 and ypS0. */
        let (yyS0new, yyS0) = {
            let m = IDA_mem.borrow();
            (m.ida_yyS0new.clone(), m.ida_yyS0.clone())
        };
        for is in 0..(Ns as usize) {
            N_VScale(ONE, &yyS0new[is], &yyS0[is]);
        }
    }

    if IDA_mem.borrow().ida_icopt == IDA_YA_YDP_INIT {
        let (ypnew, yp0) = {
            let m = IDA_mem.borrow();
            (
                m.ida_ypnew.clone().expect("ida_ypnew"),
                m.ida_yp0.clone().expect("ida_yp0"),
            )
        };
        N_VScale(ONE, &ypnew, &yp0);

        if sensi_sim {
            let (ypS0new, ypS0) = {
                let m = IDA_mem.borrow();
                (m.ida_ypS0new.clone(), m.ida_ypS0.clone())
            };
            for is in 0..(Ns as usize) {
                N_VScale(ONE, &ypS0new[is], &ypS0[is]);
            }
        }
    }
    /* Update fnorm, then return. */
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

    /* Compute the WRMS-norm. */
    *fnorm = IDAWrmsNorm(IDA_mem, &delnew, &ewt, SUNFALSE);

    /* Are we computing SENSITIVITIES with the IDA_SIMULTANEOUS approach? */

    let (sensi_sim, Ns) = {
        let m = IDA_mem.borrow();
        (m.ida_sensi && (m.ida_ism == IDA_SIMULTANEOUS), m.ida_Ns)
    };
    if sensi_sim {
        /* Evaluate the residual for sensitivities. */
        let (resS, yyS0new, ypS0new, delnewS, savresS, ewtS, tmpS1, tmpS2, tmpS3) = {
            let m = IDA_mem.borrow();
            (
                m.ida_resS.expect("ida_resS"),
                m.ida_yyS0new.clone(),
                m.ida_ypS0new.clone(),
                m.ida_delnewS.clone(),
                m.ida_savresS.clone(),
                m.ida_ewtS.clone(),
                m.ida_tmpS1.clone().expect("ida_tmpS1"),
                m.ida_tmpS2.clone().expect("ida_tmpS2"),
                m.ida_tmpS3.clone().expect("ida_tmpS3"),
            )
        };
        let mut user_dataS = IDA_mem.borrow_mut().ida_user_dataS.take();
        /* C: `ida_user_dataS` is `IDA_mem` when the internal DQ residual is in
        use and `ida_user_data` otherwise (idas.c:1359/1365). Invariant D:
        `Some(box)` is the module-owned token, `None` means hand over
        `ida_user_data`. */
        let resS_from_user_data = user_dataS.is_none();
        if resS_from_user_data {
            user_dataS = IDA_mem.borrow_mut().ida_user_data.take();
        }
        retval = resS(
            Ns,
            t0,
            &ynew,
            &ypnew,
            &savres,
            &yyS0new[..],
            &ypS0new[..],
            &delnewS[..],
            &mut user_dataS,
            &tmpS1,
            &tmpS2,
            &tmpS3,
        );
        {
            let mut m = IDA_mem.borrow_mut();
            if resS_from_user_data {
                m.ida_user_data = user_dataS;
            } else {
                m.ida_user_dataS = user_dataS;
            }
            m.ida_nrSe += 1;
        }
        if retval < 0 {
            return IDA_RES_FAIL;
        }
        if retval > 0 {
            return IC_FAIL_RECOV;
        }

        /* Save delnewS in savresS. */
        for is in 0..(Ns as usize) {
            N_VScale(ONE, &delnewS[is], &savresS[is]);
        }

        /* Call the linear solve function to get J-inverse deltaS. */
        for is in 0..(Ns as usize) {
            retval = lsolve(IDA_mem, &delnewS[is], &ewtS[is], &ynew, &ypnew, &savres);
            if retval < 0 {
                return IDA_LSOLVE_FAIL;
            }
            if retval > 0 {
                return IC_FAIL_RECOV;
            }
        }

        /* Include sensitivities in norm. */
        *fnorm = IDASensWrmsNormUpdate(IDA_mem, *fnorm, &delnewS[..], &ewtS[..], SUNFALSE);
    }

    /* Rescale norm if index = 0. */
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
    let mut retval: i32;

    retval = IDA_SUCCESS;

    /* IDA_YA_YDP_INIT case: ynew  = yy0 - lambda*delta    where id_i = 0
    ypnew = yp0 - cj*lambda*delta where id_i = 1. */
    let icopt = IDA_mem.borrow().ida_icopt;
    if icopt == IDA_YA_YDP_INIT {
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
    } else if icopt == IDA_Y_INIT {
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
    }

    let sensi_sim = {
        let m = IDA_mem.borrow();
        m.ida_sensi && (m.ida_ism == IDA_SIMULTANEOUS)
    };
    if sensi_sim {
        retval = IDASensNewyyp(IDA_mem, lambda);
    }

    retval
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
 * Sensitivity I.C. functions
 * -----------------------------------------------------------------
 */

/*
 * -----------------------------------------------------------------
 * IDASensNlsIC
 * -----------------------------------------------------------------
 * IDASensNlsIC solves nonlinear systems for sensitivities consistent
 * initial conditions.  It mainly relies on IDASensNewtonIC.
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
fn IDASensNlsIC(IDA_mem: &IDAMem) -> i32 {
    let mut retval: i32;

    let (resS, Ns, t0, yy0, yp0, delta, yyS0, ypS0, deltaS, savresS, tmpS1, tmpS2, tmpS3) = {
        let m = IDA_mem.borrow();
        (
            m.ida_resS.expect("ida_resS"),
            m.ida_Ns,
            m.ida_t0,
            m.ida_yy0.clone().expect("ida_yy0"),
            m.ida_yp0.clone().expect("ida_yp0"),
            m.ida_delta.clone().expect("ida_delta"),
            m.ida_yyS0.clone(),
            m.ida_ypS0.clone(),
            m.ida_deltaS.clone(),
            m.ida_savresS.clone(),
            m.ida_tmpS1.clone().expect("ida_tmpS1"),
            m.ida_tmpS2.clone().expect("ida_tmpS2"),
            m.ida_tmpS3.clone().expect("ida_tmpS3"),
        )
    };

    let mut user_dataS = IDA_mem.borrow_mut().ida_user_dataS.take();
    /* C: `ida_user_dataS` is `IDA_mem` when the internal DQ residual is in
    use and `ida_user_data` otherwise (idas.c:1359/1365). Invariant D:
    `Some(box)` is the module-owned token, `None` means hand over
    `ida_user_data`. */
    let resS_from_user_data = user_dataS.is_none();
    if resS_from_user_data {
        user_dataS = IDA_mem.borrow_mut().ida_user_data.take();
    }
    retval = resS(
        Ns,
        t0,
        &yy0,
        &yp0,
        &delta,
        &yyS0[..],
        &ypS0[..],
        &deltaS[..],
        &mut user_dataS,
        &tmpS1,
        &tmpS2,
        &tmpS3,
    );
    {
        let mut m = IDA_mem.borrow_mut();
        if resS_from_user_data {
            m.ida_user_data = user_dataS;
        } else {
            m.ida_user_dataS = user_dataS;
        }
        m.ida_nrSe += 1;
    }
    if retval < 0 {
        return IDA_RES_FAIL;
    }
    if retval > 0 {
        return IDA_FIRST_RES_FAIL;
    }

    /* Save deltaS */
    for is in 0..(Ns as usize) {
        N_VScale(ONE, &deltaS[is], &savresS[is]);
    }

    /* Loop over nj = number of linear solve Jacobian setups. */

    for nj in 1..=2 {
        /* Call the Newton iteration routine */
        retval = IDASensNewtonIC(IDA_mem);
        if retval == IDA_SUCCESS {
            return IDA_SUCCESS;
        }

        /* If converging slowly and lsetup is nontrivial and this is the first pass,
        update Jacobian and retry. */
        let lsetup = IDA_mem.borrow().ida_lsetup;
        if retval == IC_SLOW_CONVRG && lsetup.is_some() && nj == 1 {
            /* Restore deltaS. */
            for is in 0..(Ns as usize) {
                N_VScale(ONE, &savresS[is], &deltaS[is]);
            }

            IDA_mem.borrow_mut().ida_nsetupsS += 1;
            let lsetup = lsetup.expect("ida_lsetup");
            retval = lsetup(IDA_mem, &yy0, &yp0, &delta, &tmpS1, &tmpS2, &tmpS3);
            if retval < 0 {
                return IDA_LSETUP_FAIL;
            }
            if retval > 0 {
                return IC_FAIL_RECOV;
            }

            continue;
        } else {
            return retval;
        }
    }

    IDA_SUCCESS
}

/*
 * -----------------------------------------------------------------
 * IDASensNewtonIC
 * -----------------------------------------------------------------
 * IDANewtonIC performs the Newton iteration to solve for
 * sensitivities consistent initial conditions.  It calls
 * IDASensLineSrch within each iteration.
 * On return, savresS contains the current residual vectors.
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
fn IDASensNewtonIC(IDA_mem: &IDAMem) -> i32 {
    let mut retval: i32;
    let mut fnorm: sunrealtype;
    let fnorm0: sunrealtype;
    let mut rate: sunrealtype;

    let (lsolve, Ns, deltaS, ewtS, yy0, yp0, delta) = {
        let m = IDA_mem.borrow();
        (
            m.ida_lsolve.expect("ida_lsolve"),
            m.ida_Ns,
            m.ida_deltaS.clone(),
            m.ida_ewtS.clone(),
            m.ida_yy0.clone().expect("ida_yy0"),
            m.ida_yp0.clone().expect("ida_yp0"),
            m.ida_delta.clone().expect("ida_delta"),
        )
    };

    for is in 0..(Ns as usize) {
        /* Call the linear solve function to get the Newton step, delta. */
        retval = lsolve(IDA_mem, &deltaS[is], &ewtS[is], &yy0, &yp0, &delta);
        if retval < 0 {
            return IDA_LSOLVE_FAIL;
        }
        if retval > 0 {
            return IC_FAIL_RECOV;
        }
    }
    /* Compute the norm of the step and return if it is small enough */
    fnorm = IDASensWrmsNorm(IDA_mem, &deltaS[..], &ewtS[..], SUNFALSE);
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

    rate = ZERO;

    /* Newton iteration loop */
    let maxnit = IDA_mem.borrow().ida_maxnit;
    for _mnewt in 0..maxnit {
        IDA_mem.borrow_mut().ida_nniS += 1;
        let mut delnorm = fnorm;
        let oldfnrm = fnorm;

        /* Call the Linesearch function and return if it failed. */
        retval = IDASensLineSrch(IDA_mem, &mut delnorm, &mut fnorm);
        if retval != IDA_SUCCESS {
            return retval;
        }

        /* Set the observed convergence rate and test for convergence. */
        rate = fnorm / oldfnrm;
        if fnorm <= IDA_mem.borrow().ida_epsNewt {
            return IDA_SUCCESS;
        }

        /* If not converged, copy new step vectors, and loop. */
        let delnewS = { IDA_mem.borrow().ida_delnewS.clone() };
        for is in 0..(Ns as usize) {
            N_VScale(ONE, &delnewS[is], &deltaS[is]);
        }
    } /* End of Newton iteration loop */

    /* Return either IC_SLOW_CONVRG or recoverable fail flag. */
    if rate <= ICRATEMAX || fnorm < PT1 * fnorm0 {
        return IC_SLOW_CONVRG;
    }
    IC_CONV_FAIL
}

/*
 * -----------------------------------------------------------------
 * IDASensLineSrch
 * -----------------------------------------------------------------
 * IDASensLineSrch performs the Linesearch algorithm with the
 * calculation of consistent initial conditions for sensitivities
 * systems.
 *
 * On entry, yyS0 and ypS0 contain the current values, the Newton
 * steps are contained in deltaS, the current residual vectors FS are
 * savresS, delnorm is sens-WRMS-norm(deltaS), and fnorm is
 * max { WRMS-norm( J-inverse FS[is] ) : is=1,2,...,Ns }
 *
 * On a successful return, yy0, yp0, and savres have been updated,
 * delnew contains the current values of J-inverse FS, and fnorm is
 * max { WRMS-norm(delnewS[is]) : is = 1,2,...Ns }
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

fn IDASensLineSrch(IDA_mem: &IDAMem, delnorm: &mut sunrealtype, fnorm: &mut sunrealtype) -> i32 {
    let mut nbacks: i32;
    let f1norm: sunrealtype;
    let mut fnormp: sunrealtype = ZERO;
    let slpi: sunrealtype;
    let minlam: sunrealtype;
    let mut lambda: sunrealtype;
    let ratio: sunrealtype;

    /* Set work space pointer. */
    let phi3 = { IDA_mem.borrow().ida_phi[3].clone() }.expect("ida_phi[3]");
    IDA_mem.borrow_mut().ida_dtemp = Some(phi3);

    f1norm = (*fnorm) * (*fnorm) * HALF;

    /* Initialize local variables. */
    ratio = ONE;
    slpi = -TWO * f1norm * ratio;
    minlam = IDA_mem.borrow().ida_steptol / (*delnorm);
    lambda = ONE;
    nbacks = 0;

    let Ns = IDA_mem.borrow().ida_Ns;

    loop {
        if nbacks == IDA_mem.borrow().ida_maxbacks {
            return IC_LINESRCH_FAILED;
        }
        /* Get new iteration in (ySnew, ypSnew). */
        IDASensNewyyp(IDA_mem, lambda);

        /* Get the norm of new function value. */
        let retval = IDASensfnorm(IDA_mem, &mut fnormp);
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
    }

    /* Update yyS0, ypS0 and fnorm and return. */
    let (yyS0new, yyS0) = {
        let m = IDA_mem.borrow();
        (m.ida_yyS0new.clone(), m.ida_yyS0.clone())
    };
    for is in 0..(Ns as usize) {
        N_VScale(ONE, &yyS0new[is], &yyS0[is]);
    }

    if IDA_mem.borrow().ida_icopt == IDA_YA_YDP_INIT {
        let (ypS0new, ypS0) = {
            let m = IDA_mem.borrow();
            (m.ida_ypS0new.clone(), m.ida_ypS0.clone())
        };
        for is in 0..(Ns as usize) {
            N_VScale(ONE, &ypS0new[is], &ypS0[is]);
        }
    }

    *fnorm = fnormp;
    IDA_SUCCESS
}

/*
 * -----------------------------------------------------------------
 * IDASensfnorm
 * -----------------------------------------------------------------
 * IDASensfnorm computes the norm of the current function value, by
 * evaluating the sensitivity residual function, calling the linear
 * system solver, and computing a WRMS-norm.
 *
 * On return, savresS contains the current residual vectors FS, and
 * delnewS contains J-inverse FS.
 *
 * The return value is IDA_SUCCESS = 0 if no error occurred, or
 *  IC_FAIL_RECOV    if res or lsolve failed recoverably, or
 *  IDA_RES_FAIL     if res had a non-recoverable error, or
 *  IDA_LSOLVE_FAIL  if lsolve had a non-recoverable error.
 * -----------------------------------------------------------------
 */

fn IDASensfnorm(IDA_mem: &IDAMem, fnorm: &mut sunrealtype) -> i32 {
    let mut retval: i32;

    /* Get sensitivity residual */
    let (
        resS,
        lsolve,
        Ns,
        t0,
        yy0,
        yp0,
        delta,
        yyS0new,
        ypS0new,
        delnewS,
        savresS,
        ewtS,
        tmpS1,
        tmpS2,
        tmpS3,
    ) = {
        let m = IDA_mem.borrow();
        (
            m.ida_resS.expect("ida_resS"),
            m.ida_lsolve.expect("ida_lsolve"),
            m.ida_Ns,
            m.ida_t0,
            m.ida_yy0.clone().expect("ida_yy0"),
            m.ida_yp0.clone().expect("ida_yp0"),
            m.ida_delta.clone().expect("ida_delta"),
            m.ida_yyS0new.clone(),
            m.ida_ypS0new.clone(),
            m.ida_delnewS.clone(),
            m.ida_savresS.clone(),
            m.ida_ewtS.clone(),
            m.ida_tmpS1.clone().expect("ida_tmpS1"),
            m.ida_tmpS2.clone().expect("ida_tmpS2"),
            m.ida_tmpS3.clone().expect("ida_tmpS3"),
        )
    };

    let mut user_dataS = IDA_mem.borrow_mut().ida_user_dataS.take();
    /* C: `ida_user_dataS` is `IDA_mem` when the internal DQ residual is in
    use and `ida_user_data` otherwise (idas.c:1359/1365). Invariant D:
    `Some(box)` is the module-owned token, `None` means hand over
    `ida_user_data`. */
    let resS_from_user_data = user_dataS.is_none();
    if resS_from_user_data {
        user_dataS = IDA_mem.borrow_mut().ida_user_data.take();
    }
    retval = resS(
        Ns,
        t0,
        &yy0,
        &yp0,
        &delta,
        &yyS0new[..],
        &ypS0new[..],
        &delnewS[..],
        &mut user_dataS,
        &tmpS1,
        &tmpS2,
        &tmpS3,
    );
    {
        let mut m = IDA_mem.borrow_mut();
        if resS_from_user_data {
            m.ida_user_data = user_dataS;
        } else {
            m.ida_user_dataS = user_dataS;
        }
        m.ida_nrSe += 1;
    }
    if retval < 0 {
        return IDA_RES_FAIL;
    }
    if retval > 0 {
        return IC_FAIL_RECOV;
    }

    for is in 0..(Ns as usize) {
        N_VScale(ONE, &delnewS[is], &savresS[is]);
    }

    /* Call linear solve function */
    for is in 0..(Ns as usize) {
        retval = lsolve(IDA_mem, &delnewS[is], &ewtS[is], &yy0, &yp0, &delta);
        if retval < 0 {
            return IDA_LSOLVE_FAIL;
        }
        if retval > 0 {
            return IC_FAIL_RECOV;
        }
    }

    /* Compute the WRMS-norm; rescale if index = 0. */
    *fnorm = IDASensWrmsNorm(IDA_mem, &delnewS[..], &ewtS[..], SUNFALSE);
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
 * IDASensNewyyp
 * -----------------------------------------------------------------
 * IDASensNewyyp computes the Newton updates for each of the
 * sensitivities systems using the current step vector lambda*delta,
 * in a manner depending on icopt and the input id vector.
 *
 * The return value is always IDA_SUCCESS = 0.
 * -----------------------------------------------------------------
 */

fn IDASensNewyyp(IDA_mem: &IDAMem, lambda: sunrealtype) -> i32 {
    let (icopt, Ns, deltaS, dtemp, ypS0, ypS0new, yyS0, yyS0new, cj, id) = {
        let m = IDA_mem.borrow();
        (
            m.ida_icopt,
            m.ida_Ns,
            m.ida_deltaS.clone(),
            m.ida_dtemp.clone(),
            m.ida_ypS0.clone(),
            m.ida_ypS0new.clone(),
            m.ida_yyS0.clone(),
            m.ida_yyS0new.clone(),
            m.ida_cj,
            m.ida_id.clone(),
        )
    };

    if icopt == IDA_YA_YDP_INIT {
        /* IDA_YA_YDP_INIT case:
        - ySnew  = yS0  - lambda*deltaS    where id_i = 0
        - ypSnew = ypS0 - cj*lambda*delta  where id_i = 1. */

        let id = id.expect("ida_id");
        let dtemp = dtemp.expect("ida_dtemp");
        for is in 0..(Ns as usize) {
            /* It is ok to use dtemp as temporary vector here. */
            N_VProd(&id, &deltaS[is], &dtemp);
            N_VLinearSum(ONE, &ypS0[is], -cj * lambda, &dtemp, &ypS0new[is]);
            N_VLinearSum(ONE, &deltaS[is], -ONE, &dtemp, &dtemp);
            N_VLinearSum(ONE, &yyS0[is], -lambda, &dtemp, &yyS0new[is]);
        } /* end loop is */
    } else {
        /* IDA_Y_INIT case:
        - ySnew = yS0 - lambda*deltaS. (ypnew = yp0 preset.) */

        for is in 0..(Ns as usize) {
            N_VLinearSum(ONE, &yyS0[is], -lambda, &deltaS[is], &yyS0new[is]);
        }
    } /* end loop is */
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
