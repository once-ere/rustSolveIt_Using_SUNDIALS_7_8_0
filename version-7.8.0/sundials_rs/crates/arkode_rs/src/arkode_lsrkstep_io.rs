//! Port of `src/arkode/arkode_lsrkstep_io.c` (optional input and output
//! functions for the ARKODE LSRKStep time stepper module). The stepper
//! itself, its content record, and `lsrkStep_mem_mut` live in
//! `arkode_lsrkstep.rs`.
//!
//! Binding notes:
//!  * The C key tables hold pointers to the public `LSRKStepSet*` routines,
//!    which receive the raw `void* arkode_mem` forwarded by the
//!    `sunCheckAndSet*Args` helpers. Here each table entry is a small
//!    adapter matching `sundials_core::sundials_cli`'s setter fn types: it
//!    downcasts the token (an `Option<Box<dyn Any>>` holding an `ARKodeMem`
//!    clone) back to the handle and forwards to the real setter. The
//!    `sunbooleantype` setter is fed an `int` straight from `atoi` in C, so
//!    the adapter converts with `arg != 0`.
//!  * `lsrkStep_mem_mut` guards are never held across `arkProcessError`,
//!    an N_Vector operation, a `SUNDomEigEstimator` call, or another borrow
//!    of the same mem.
//!  * The `switch` `default:` arms over `ARKODE_LSRKMethodType` and over
//!    `sunbooleantype` are unreachable in Rust (the enum has exactly the
//!    five upstream values, `bool` exactly two) and are therefore dropped.

use std::any::Any;

use sundials_core::sundials_cli::*;
use sundials_core::sundials_domeigestimator::{SUNDomEigEstimator, SUNDomEigEstimator_SetATimes};
use sundials_core::sundials_errors::SUN_SUCCESS;
use sundials_core::sundials_linearsolver::SUNATimesFn;
use sundials_core::sundials_nvector::{N_VScale, N_Vector};
use sundials_core::sundials_types::*;
use sundials_core::sundials_utils::{sun_format_g, sunfprintf_long, sunfprintf_real, SUNFile};

use crate::arkode_impl::*;
use crate::arkode_io::arkReplaceAdaptController;
use crate::arkode_lsrkstep::*;

/*===============================================================
  Exported optional input functions.
  ===============================================================*/

/*---------------------------------------------------------------
  LSRKStepSetSTSMethod sets method
    ARKODE_LSRK_RKC_2
    ARKODE_LSRK_RKL_2
  ---------------------------------------------------------------*/
pub fn LSRKStepSetSTSMethod(arkode_mem: &ARKodeMem, method: ARKODE_LSRKMethodType) -> i32 {
    /* access ARKodeMem and ARKodeLSRKStepMem structures */
    let retval = lsrkStep_AccessARKODEStepMem(arkode_mem, "LSRKStepSetSTSMethod");
    if retval != ARK_SUCCESS {
        return retval;
    }
    let ark_mem = arkode_mem;

    match method {
        ARKODE_LSRK_RKC_2 => {
            ark_mem.borrow_mut().step = Some(lsrkStep_TakeStepRKC);
            {
                let mut step_mem = lsrkStep_mem_mut(ark_mem);
                step_mem.is_SSP = SUNFALSE;
                step_mem.nfusedopvecs = 5;
                step_mem.q = 2;
                step_mem.p = 2;
                step_mem.step_nst = 0;
            }
            {
                let mut m = ark_mem.borrow_mut();
                let hadapt_mem = m.hadapt_mem.as_mut().expect("hadapt_mem");
                hadapt_mem.q = 2;
                hadapt_mem.p = 2;
            }
        }
        ARKODE_LSRK_RKL_2 => {
            ark_mem.borrow_mut().step = Some(lsrkStep_TakeStepRKL);
            {
                let mut step_mem = lsrkStep_mem_mut(ark_mem);
                step_mem.is_SSP = SUNFALSE;
                step_mem.nfusedopvecs = 5;
                step_mem.q = 2;
                step_mem.p = 2;
                step_mem.step_nst = 0;
            }
            {
                let mut m = ark_mem.borrow_mut();
                let hadapt_mem = m.hadapt_mem.as_mut().expect("hadapt_mem");
                hadapt_mem.q = 2;
                hadapt_mem.p = 2;
            }
        }
        ARKODE_LSRK_SSP_S_2 | ARKODE_LSRK_SSP_S_3 | ARKODE_LSRK_SSP_10_4 => {
            /* C `break`s out of the switch here (no early return), so the
               method is still recorded and ARK_SUCCESS is returned. */
            arkProcessError(
                Some(ark_mem),
                ARK_ILL_INPUT,
                line!() as i32,
                "LSRKStepSetSTSMethod",
                file!(),
                "Invalid method option: Call LSRKStepCreateSSP to create an SSP method first.",
            );
        }
    }

    lsrkStep_mem_mut(ark_mem).LSRKmethod = method;

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  LSRKStepSetSSPMethod sets method
    ARKODE_LSRK_SSP_S_2
    ARKODE_LSRK_SSP_S_3
    ARKODE_LSRK_SSP_10_4
  ---------------------------------------------------------------*/
pub fn LSRKStepSetSSPMethod(arkode_mem: &ARKodeMem, method: ARKODE_LSRKMethodType) -> i32 {
    /* access ARKodeMem and ARKodeLSRKStepMem structures */
    let retval = lsrkStep_AccessARKODEStepMem(arkode_mem, "LSRKStepSetSSPMethod");
    if retval != ARK_SUCCESS {
        return retval;
    }
    let ark_mem = arkode_mem;

    match method {
        ARKODE_LSRK_RKC_2 | ARKODE_LSRK_RKL_2 => {
            /* C `break`s out of the switch here (no early return). */
            arkProcessError(
                Some(ark_mem),
                ARK_ILL_INPUT,
                line!() as i32,
                "LSRKStepSetSSPMethod",
                file!(),
                "Invalid method option: Call LSRKStepCreateSTS to create an STS method first.",
            );
        }
        ARKODE_LSRK_SSP_S_2 => {
            ark_mem.borrow_mut().step = Some(lsrkStep_TakeStepSSPs2);
            {
                let mut step_mem = lsrkStep_mem_mut(ark_mem);
                step_mem.is_SSP = SUNTRUE;
                step_mem.req_stages = 2;
                step_mem.nfusedopvecs = 3;
                step_mem.q = 2;
                step_mem.p = 1;
            }
            {
                let mut m = ark_mem.borrow_mut();
                let hadapt_mem = m.hadapt_mem.as_mut().expect("hadapt_mem");
                hadapt_mem.q = 2;
                hadapt_mem.p = 1;
            }
        }
        ARKODE_LSRK_SSP_S_3 => {
            ark_mem.borrow_mut().step = Some(lsrkStep_TakeStepSSP43);
            {
                let mut step_mem = lsrkStep_mem_mut(ark_mem);
                step_mem.is_SSP = SUNTRUE;
                step_mem.req_stages = 4;
                step_mem.nfusedopvecs = 3;
                step_mem.q = 3;
                step_mem.p = 2;
            }
            {
                let mut m = ark_mem.borrow_mut();
                let hadapt_mem = m.hadapt_mem.as_mut().expect("hadapt_mem");
                hadapt_mem.q = 3;
                hadapt_mem.p = 2;
            }
        }
        ARKODE_LSRK_SSP_10_4 => {
            ark_mem.borrow_mut().step = Some(lsrkStep_TakeStepSSP104);
            {
                let mut step_mem = lsrkStep_mem_mut(ark_mem);
                step_mem.is_SSP = SUNTRUE;
                step_mem.req_stages = 10;
                step_mem.nfusedopvecs = 3;
                step_mem.q = 4;
                step_mem.p = 3;
            }
            {
                let mut m = ark_mem.borrow_mut();
                let hadapt_mem = m.hadapt_mem.as_mut().expect("hadapt_mem");
                hadapt_mem.q = 4;
                hadapt_mem.p = 3;
            }
        }
    }

    lsrkStep_mem_mut(ark_mem).LSRKmethod = method;

    ARK_SUCCESS
}

pub fn LSRKStepSetSTSMethodByName(arkode_mem: &ARKodeMem, emethod: &str) -> i32 {
    if emethod == "ARKODE_LSRK_RKC_2" {
        return LSRKStepSetSTSMethod(arkode_mem, ARKODE_LSRK_RKC_2);
    }
    if emethod == "ARKODE_LSRK_RKL_2" {
        return LSRKStepSetSTSMethod(arkode_mem, ARKODE_LSRK_RKL_2);
    }
    if (emethod == "ARKODE_LSRK_SSP_S_2")
        || (emethod == "ARKODE_LSRK_SSP_S_3")
        || (emethod == "ARKODE_LSRK_SSP_10_4")
    {
        /* C does not return here; it falls through to the message below. */
        arkProcessError(
            None,
            ARK_ILL_INPUT,
            line!() as i32,
            "LSRKStepSetSTSMethodByName",
            file!(),
            "Invalid method option: Call LSRKStepCreateSTS to create an STS method first.",
        );
    }

    arkProcessError(
        None,
        ARK_ILL_INPUT,
        line!() as i32,
        "LSRKStepSetSTSMethodByName",
        file!(),
        "Unknown method type",
    );

    ARK_ILL_INPUT
}

pub fn LSRKStepSetSSPMethodByName(arkode_mem: &ARKodeMem, emethod: &str) -> i32 {
    if (emethod == "ARKODE_LSRK_RKC_2") || (emethod == "ARKODE_LSRK_RKL_2") {
        /* C does not return here; it falls through to the checks below. */
        arkProcessError(
            None,
            ARK_ILL_INPUT,
            line!() as i32,
            "LSRKStepSetSSPMethodByName",
            file!(),
            "Invalid method option: Call LSRKStepCreateSSP to create an SSP method first.",
        );
    }
    if emethod == "ARKODE_LSRK_SSP_S_2" {
        return LSRKStepSetSSPMethod(arkode_mem, ARKODE_LSRK_SSP_S_2);
    }
    if emethod == "ARKODE_LSRK_SSP_S_3" {
        return LSRKStepSetSSPMethod(arkode_mem, ARKODE_LSRK_SSP_S_3);
    }
    if emethod == "ARKODE_LSRK_SSP_10_4" {
        return LSRKStepSetSSPMethod(arkode_mem, ARKODE_LSRK_SSP_10_4);
    }

    arkProcessError(
        None,
        ARK_ILL_INPUT,
        line!() as i32,
        "LSRKStepSetSSPMethodByName",
        file!(),
        "Unknown method type",
    );

    ARK_ILL_INPUT
}

/*---------------------------------------------------------------
  LSRKStepSetDomEigFn specifies the dom_eig function.
  Specifies the dominant eigenvalue approximation routine to be used for determining
  the number of stages that will be used by either the RKC or RKL methods.
  ---------------------------------------------------------------*/
pub fn LSRKStepSetDomEigFn(arkode_mem: &ARKodeMem, dom_eig: Option<ARKDomEigFn>) -> i32 {
    /* access ARKodeMem and ARKodeLSRKStepMem structures */
    let retval = lsrkStep_AccessARKODEStepMem(arkode_mem, "LSRKStepSetDomEigFn");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* set the dom_eig routine pointer, and update relevant flags */
    lsrkStep_mem_mut(arkode_mem).dom_eig_fn = dom_eig;

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  LSRKStepSetDomEigFrequency sets dom_eig computation frequency -
  Dominated Eigenvalue is recomputed after "nsteps" successful steps.

  nsteps = 0 refers to constant dominant eigenvalue
  nsteps < 0 resets the default value 25 and sets nonconstant dominant eigenvalue
  ---------------------------------------------------------------*/
pub fn LSRKStepSetDomEigFrequency(arkode_mem: &ARKodeMem, nsteps: i64) -> i32 {
    /* access ARKodeMem and ARKodeLSRKStepMem structures */
    let retval = lsrkStep_AccessARKODEStepMem(arkode_mem, "LSRKStepSetDomEigFrequency");
    if retval != ARK_SUCCESS {
        return retval;
    }

    let mut step_mem = lsrkStep_mem_mut(arkode_mem);

    if nsteps < 0 {
        step_mem.dom_eig_freq = DOM_EIG_FREQ_DEFAULT;
        step_mem.const_Jac = SUNFALSE;
    }

    /* C's second `if` is not chained to the first, so a negative nsteps
       falls into this `else` and overwrites dom_eig_freq with nsteps. */
    if nsteps == 0 {
        step_mem.const_Jac = SUNTRUE;
        step_mem.dom_eig_freq = 1;
    } else {
        step_mem.dom_eig_freq = nsteps;
        step_mem.const_Jac = SUNFALSE;
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  LSRKStepSetMaxNumStages sets the maximum number of stages allowed.
  If the combination of the maximum number of stages and the current
  time step size in the LSRKStep module does not allow for a stable
  step, the step routine returns to ARKODE for an updated (refined)
  step size. The number of such returns is tracked in a counter,
  which can be accessed using ARKodeGetNumExpSteps.
  ---------------------------------------------------------------*/
pub fn LSRKStepSetMaxNumStages(arkode_mem: &ARKodeMem, stage_max_limit: i32) -> i32 {
    /* access ARKodeMem and ARKodeLSRKStepMem structures */
    let retval = lsrkStep_AccessARKODEStepMem(arkode_mem, "LSRKStepSetMaxNumStages");
    if retval != ARK_SUCCESS {
        return retval;
    }

    let mut step_mem = lsrkStep_mem_mut(arkode_mem);
    if stage_max_limit < 2 {
        step_mem.stage_max_limit = STAGE_MAX_LIMIT_DEFAULT;
    } else {
        step_mem.stage_max_limit = stage_max_limit;
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  LSRKStepSetDomEigSafetyFactor sets the safety factor for the DomEigs.
  Specifies a safety factor to use for the result of the dominant eigenvalue estimation function.
  This value is used to scale the magnitude of the dominant eigenvalue, in the hope of ensuring
  a sufficient number of stages for the method to be stable.  This input is only used for RKC
  and RKL methods.

  Calling this function with dom_eig_safety < 0 resets the default value
  ---------------------------------------------------------------*/
pub fn LSRKStepSetDomEigSafetyFactor(arkode_mem: &ARKodeMem, dom_eig_safety: sunrealtype) -> i32 {
    /* access ARKodeMem and ARKodeLSRKStepMem structures */
    let retval = lsrkStep_AccessARKODEStepMem(arkode_mem, "LSRKStepSetDomEigSafetyFactor");
    if retval != ARK_SUCCESS {
        return retval;
    }

    let mut step_mem = lsrkStep_mem_mut(arkode_mem);
    if dom_eig_safety < 1.0 {
        step_mem.dom_eig_safety = DOM_EIG_SAFETY_DEFAULT;
    } else {
        step_mem.dom_eig_safety = dom_eig_safety;
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  LSRKStepSetUseAnalyticStabilityRegion sets whether to use the ellipse or the exact
  stability region for stability checks.
  ---------------------------------------------------------------*/
pub fn LSRKStepSetUseAnalyticStabilityRegion(
    arkode_mem: &ARKodeMem,
    use_analytic_stab_region: sunbooleantype,
) -> i32 {
    /* access ARKodeMem and ARKodeLSRKStepMem structures */
    let retval = lsrkStep_AccessARKODEStepMem(arkode_mem, "LSRKStepSetUseAnalyticStabilityRegion");
    if retval != ARK_SUCCESS {
        return retval;
    }

    let mut step_mem = lsrkStep_mem_mut(arkode_mem);
    step_mem.dom_eig_update = SUNTRUE;
    step_mem.dom_eig_is_current = SUNFALSE;
    step_mem.use_ellipse = !use_analytic_stab_region;

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  LSRKStepSetNumDomEigEstInitPreprocessIters sets the number of the preprocessing
  iterations before the very first estimate call.
  ---------------------------------------------------------------*/
pub fn LSRKStepSetNumDomEigEstInitPreprocessIters(
    arkode_mem: &ARKodeMem,
    num_iters: i32,
) -> i32 {
    /* access ARKodeMem and ARKodeLSRKStepMem structures */
    let retval =
        lsrkStep_AccessARKODEStepMem(arkode_mem, "LSRKStepSetNumDomEigEstInitPreprocessIters");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* This value will be used in lsrkStep_Init to set the number of preprocessing
       iterations for the first dominant eigenvalue estimate. If num_iters < 0,
       then the DEE's default will be used. */
    lsrkStep_mem_mut(arkode_mem).num_init_warmups = num_iters;

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  LSRKStepSetNumDomEigEstPreprocessIters sets the number of the preprocessing
  iterations before each estimate call after the initial estimate call.
  ---------------------------------------------------------------*/
pub fn LSRKStepSetNumDomEigEstPreprocessIters(arkode_mem: &ARKodeMem, num_iters: i32) -> i32 {
    /* access ARKodeMem and ARKodeLSRKStepMem structures */
    let retval = lsrkStep_AccessARKODEStepMem(arkode_mem, "LSRKStepSetNumDomEigEstPreprocessIters");
    if retval != ARK_SUCCESS {
        return retval;
    }
    let ark_mem = arkode_mem;

    {
        let mut step_mem = lsrkStep_mem_mut(ark_mem);
        if num_iters < 0 {
            step_mem.num_warmups = DOM_EIG_NUM_WARMUPS_DEFAULT;
        } else {
            step_mem.num_warmups = num_iters;
        }
    }

    /* Set the number of iterations immediately (if possible) to allow the user to
       can change the value at any time during the integration. This value will be
       overridden for the first estimate by the value set with InitPreprocessIters
       above then reset to the supplied value afterward.

       A (perhaps pathological) corner case can occur where this value is not
       applied if a user detaches the DEE, calls this function then attaches a new
       DEE (or reattached the same DEE), and reinit is not called before
       evolve. In that case, whatever value the DEE has will be used. This could
       be avoided (with additional overhead) by calling SetNumPreprocessIters
       before every Estimate call. */
    let (DEE, num_init_warmups) = {
        let step_mem = lsrkStep_mem_mut(ark_mem);
        (step_mem.DEE.clone(), step_mem.num_init_warmups)
    };
    if let Some(DEE) = DEE {
        /* NOTE: C passes `num_init_warmups` here, not `num_warmups`. */
        let retval = sundials_core::sundials_domeigestimator::SUNDomEigEstimator_SetNumPreprocessIters(
            &DEE,
            num_init_warmups,
        );
        if retval != SUN_SUCCESS {
            arkProcessError(
                Some(ark_mem),
                ARK_DEE_FAIL,
                line!() as i32,
                "LSRKStepSetNumDomEigEstPreprocessIters",
                file!(),
                "SUNDomeEigEstimator_SetNumPreprocessIters failed",
            );
            return ARK_DEE_FAIL;
        }
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  LSRKStepSetNumSSPStages sets the number of stages in the following
  SSP methods:

      ARKODE_LSRK_SSP_S_2  -- num_of_stages must be greater than or equal to 2
      ARKODE_LSRK_SSP_S_3  -- num_of_stages must be a perfect square greater than or equal to 4
      ARKODE_LSRK_SSP_10_4 -- num_of_stages must be equal to 10 - no need to call!

   Sets the number of stages, s in SSP(s, p) methods. This input is only utilized by
   SSPRK methods. Thus, this set routine must be called after calling LSRKStepSetSSPMethod.

   Calling this function with num_of_stages =< 0 resets the default value.
  ---------------------------------------------------------------*/
pub fn LSRKStepSetNumSSPStages(arkode_mem: &ARKodeMem, num_of_stages: i32) -> i32 {
    /* access ARKodeMem and ARKodeLSRKStepMem structures */
    let retval = lsrkStep_AccessARKODEStepMem(arkode_mem, "LSRKStepSetNumSSPStages");
    if retval != ARK_SUCCESS {
        return retval;
    }
    let ark_mem = arkode_mem;

    if !lsrkStep_mem_mut(ark_mem).is_SSP {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "LSRKStepSetNumSSPStages",
            file!(),
            "Call this function only for SSP methods: Use LSRKStepSetSSPMethod to declare SSP \
             method type first!",
        );
        return ARK_ILL_INPUT;
    }

    let LSRKmethod = lsrkStep_mem_mut(ark_mem).LSRKmethod;

    if num_of_stages <= 0 {
        match LSRKmethod {
            ARKODE_LSRK_SSP_S_2 => lsrkStep_mem_mut(ark_mem).req_stages = 2,

            ARKODE_LSRK_SSP_S_3 => lsrkStep_mem_mut(ark_mem).req_stages = 4,

            ARKODE_LSRK_SSP_10_4 => lsrkStep_mem_mut(ark_mem).req_stages = 10,

            /* C's `default:` arm ("Call LSRKStepSetSSPMethod to declare SSP
               method type first!") is reachable only for the STS method
               ids, which the `is_SSP` guard above already excludes. */
            ARKODE_LSRK_RKC_2 | ARKODE_LSRK_RKL_2 => {
                arkProcessError(
                    Some(ark_mem),
                    ARK_ILL_INPUT,
                    line!() as i32,
                    "LSRKStepSetNumSSPStages",
                    file!(),
                    "Call LSRKStepSetSSPMethod to declare SSP method type first!",
                );
                return ARK_ILL_INPUT;
            }
        }
        return ARK_SUCCESS;
    } else {
        match LSRKmethod {
            ARKODE_LSRK_SSP_S_2 => {
                if num_of_stages < 2 {
                    arkProcessError(
                        Some(ark_mem),
                        ARK_ILL_INPUT,
                        line!() as i32,
                        "LSRKStepSetNumSSPStages",
                        file!(),
                        "num_of_stages must be greater than or equal to 2, or set it less than \
                         or equal to 0 to reset the default value",
                    );
                    return ARK_ILL_INPUT;
                }
            }

            ARKODE_LSRK_SSP_S_3 => {
                /* The SSP3 method differs significantly when s = 4. Therefore, the case
                where num_of_stages = 4 is considered separately to avoid unnecessary
                boolean checks and improve computational efficiency. */

                /* We check that num_of_stages is a perfect square. Note the call to sqrt
                 * rather than SUNRsqrt which could cause loss of precision if
                 * sunrealtype is float. sqrt cannot produce a number bigger than INT_MAX
                 * here so there's no problem with the cast back to int */
                let root = (num_of_stages as f64).sqrt() as i32;
                if num_of_stages < 4 || root * root != num_of_stages {
                    arkProcessError(
                        Some(ark_mem),
                        ARK_ILL_INPUT,
                        line!() as i32,
                        "LSRKStepSetNumSSPStages",
                        file!(),
                        "num_of_stages must be a perfect square greater than or equal to 4, or \
                         set it less than or equal to 0 to reset the default value",
                    );
                    return ARK_ILL_INPUT;
                }
                if num_of_stages == 4 {
                    ark_mem.borrow_mut().step = Some(lsrkStep_TakeStepSSP43);
                } else {
                    ark_mem.borrow_mut().step = Some(lsrkStep_TakeStepSSPs3);
                }
            }

            ARKODE_LSRK_SSP_10_4 => {
                if num_of_stages != 10 {
                    arkProcessError(
                        Some(ark_mem),
                        ARK_ILL_INPUT,
                        line!() as i32,
                        "LSRKStepSetNumSSPStages",
                        file!(),
                        "SSP10_4 method has a prefixed num_of_stages = 10",
                    );
                    return ARK_ILL_INPUT;
                }
            }

            ARKODE_LSRK_RKC_2 | ARKODE_LSRK_RKL_2 => {
                arkProcessError(
                    Some(ark_mem),
                    ARK_ILL_INPUT,
                    line!() as i32,
                    "LSRKStepSetNumSSPStages",
                    file!(),
                    "Call LSRKStepSetSSPMethod to declare SSP method type first!",
                );
                return ARK_ILL_INPUT;
            }
        }
        lsrkStep_mem_mut(ark_mem).req_stages = num_of_stages;
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  LSRKStepSetDomEigEstimator:

  This routine sets the dominant eigenvalue estimator DEE.
  ---------------------------------------------------------------*/
pub fn LSRKStepSetDomEigEstimator(
    arkode_mem: &ARKodeMem,
    DEE: Option<&SUNDomEigEstimator>,
) -> i32 {
    /* access ARKodeMem and ARKodeLSRKStepMem structures */
    let retval = lsrkStep_AccessARKODEStepMem(arkode_mem, "LSRKStepSetDomEigEstimator");
    if retval != ARK_SUCCESS {
        return retval;
    }
    let ark_mem = arkode_mem;

    let DEE = match DEE {
        None => {
            lsrkStep_mem_mut(ark_mem).DEE = None;
            return ARK_SUCCESS;
        }
        Some(DEE) => DEE,
    };

    /* C also checks `DEE->ops == NULL`; the ops struct is held by value. */

    if DEE.ops.borrow().estimate.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "LSRKStepSetDomEigEstimator",
            file!(),
            "Null SUNDomEigEstimator estimate operation",
        );
        return ARK_ILL_INPUT;
    }

    /* Attach the DEE pointer to the step memory */
    lsrkStep_mem_mut(ark_mem).DEE = Some(DEE.clone());

    /* Set the ATimes function for the DEE with A_data = arkode_mem */
    let A_data: Box<dyn Any> = Box::new(arkode_mem.clone());
    let retval = SUNDomEigEstimator_SetATimes(
        DEE,
        Some(A_data),
        Some(lsrkStep_DQJtimes as SUNATimesFn),
    );
    if retval != SUN_SUCCESS {
        arkProcessError(
            Some(ark_mem),
            ARK_DEE_FAIL,
            line!() as i32,
            "LSRKStepSetDomEigEstimator",
            file!(),
            "SUNDomEigEstimator_SetATimes failed",
        );
        return ARK_DEE_FAIL;
    }

    ARK_SUCCESS
}

/*===============================================================
  Exported optional output functions.
  ===============================================================*/

/*---------------------------------------------------------------
  LSRKStepGetNumDomEigUpdates:

  Returns the number of dominant eigenvalue updates
  ---------------------------------------------------------------*/
pub fn LSRKStepGetNumDomEigUpdates(arkode_mem: &ARKodeMem, dom_eig_num_evals: &mut i64) -> i32 {
    /* access ARKodeMem and ARKodeLSRKStepMem structures */
    let retval = lsrkStep_AccessARKODEStepMem(arkode_mem, "LSRKStepGetNumDomEigUpdates");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* C checks `dom_eig_num_evals == NULL`; impossible through `&mut i64` */

    /* get values from step_mem */
    *dom_eig_num_evals = lsrkStep_mem_mut(arkode_mem).dom_eig_num_evals;

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  LSRKStepGetMaxNumStages:

  Returns the max number of stages used
  ---------------------------------------------------------------*/
pub fn LSRKStepGetMaxNumStages(arkode_mem: &ARKodeMem, stage_max: &mut i32) -> i32 {
    /* access ARKodeMem and ARKodeLSRKStepMem structures */
    let retval = lsrkStep_AccessARKODEStepMem(arkode_mem, "LSRKStepGetMaxNumStages");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* C checks `stage_max == NULL`; impossible through `&mut i32` */

    /* get values from step_mem */
    *stage_max = lsrkStep_mem_mut(arkode_mem).stage_max;

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  LSRKStepGetNumDomEigEstRhsEvals:

  Returns the number of RHS evals in DQ Jacobian computations
  ---------------------------------------------------------------*/
pub fn LSRKStepGetNumDomEigEstRhsEvals(arkode_mem: &ARKodeMem, nfeDQ: &mut i64) -> i32 {
    /* access ARKodeMem and ARKodeLSRKStepMem structures */
    let retval = lsrkStep_AccessARKODEStepMem(arkode_mem, "LSRKStepGetNumDomEigEstRhsEvals");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* C checks `nfeDQ == NULL`; impossible through `&mut i64` */

    /* get values from step_mem */
    *nfeDQ = lsrkStep_mem_mut(arkode_mem).nfeDQ;

    ARK_SUCCESS
}

pub fn LSRKStepGetNumDomEigEstIters(arkode_mem: &ARKodeMem, num_iters: &mut i64) -> i32 {
    /* access ARKodeMem and ARKodeLSRKStepMem structures */
    let retval = lsrkStep_AccessARKODEStepMem(arkode_mem, "LSRKStepGetNumDomEigEstIters");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* C checks `num_iters == NULL`; impossible through `&mut i64` */

    /* get values from step_mem */
    *num_iters = lsrkStep_mem_mut(arkode_mem).num_dee_iters;

    ARK_SUCCESS
}

/*===============================================================
  Private functions attached to ARKODE
  ===============================================================*/

/* -----------------------------------------------------------------
 * Adapter helpers: recover the ARKodeMem handle from the CLI token
 * (C: the raw `void* arkode_mem` passed through sunCheckAndSet*Args).
 * A missing/mistyped token corresponds to C passing a garbage pointer
 * to the setter (UB) and maps to a deterministic panic.
 * ----------------------------------------------------------------- */

fn cliARKodeMem(mem: &mut Option<Box<dyn Any>>) -> ARKodeMem {
    mem.as_mut()
        .and_then(|b| b.downcast_ref::<ARKodeMem>())
        .cloned()
        .expect("arkode_mem token")
}

/* "char*" setter adapters */

fn cliLSRKStepSetSTSMethodByName(mem: &mut Option<Box<dyn Any>>, arg: &str) -> i32 {
    let ark_mem = cliARKodeMem(mem);
    LSRKStepSetSTSMethodByName(&ark_mem, arg)
}

fn cliLSRKStepSetSSPMethodByName(mem: &mut Option<Box<dyn Any>>, arg: &str) -> i32 {
    let ark_mem = cliARKodeMem(mem);
    LSRKStepSetSSPMethodByName(&ark_mem, arg)
}

/* "long int" setter adapters */

fn cliLSRKStepSetDomEigFrequency(mem: &mut Option<Box<dyn Any>>, arg: i64) -> i32 {
    let ark_mem = cliARKodeMem(mem);
    LSRKStepSetDomEigFrequency(&ark_mem, arg)
}

/* "int" setter adapters */

fn cliLSRKStepSetMaxNumStages(mem: &mut Option<Box<dyn Any>>, arg: i32) -> i32 {
    let ark_mem = cliARKodeMem(mem);
    LSRKStepSetMaxNumStages(&ark_mem, arg)
}

fn cliLSRKStepSetNumSSPStages(mem: &mut Option<Box<dyn Any>>, arg: i32) -> i32 {
    let ark_mem = cliARKodeMem(mem);
    LSRKStepSetNumSSPStages(&ark_mem, arg)
}

fn cliLSRKStepSetNumDomEigEstInitPreprocessIters(
    mem: &mut Option<Box<dyn Any>>,
    arg: i32,
) -> i32 {
    let ark_mem = cliARKodeMem(mem);
    LSRKStepSetNumDomEigEstInitPreprocessIters(&ark_mem, arg)
}

fn cliLSRKStepSetNumDomEigEstPreprocessIters(mem: &mut Option<Box<dyn Any>>, arg: i32) -> i32 {
    let ark_mem = cliARKodeMem(mem);
    LSRKStepSetNumDomEigEstPreprocessIters(&ark_mem, arg)
}

fn cliLSRKStepSetUseAnalyticStabilityRegion(mem: &mut Option<Box<dyn Any>>, arg: i32) -> i32 {
    let ark_mem = cliARKodeMem(mem);
    /* C passes the int straight through as sunbooleantype */
    LSRKStepSetUseAnalyticStabilityRegion(&ark_mem, arg != 0)
}

/* "sunrealtype" setter adapters */

fn cliLSRKStepSetDomEigSafetyFactor(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let ark_mem = cliARKodeMem(mem);
    LSRKStepSetDomEigSafetyFactor(&ark_mem, arg)
}

/*---------------------------------------------------------------
  lsrkStep_SetOption:

  Provides string-based control over LSRKStep-specific "set" routines.
  ---------------------------------------------------------------*/
pub fn lsrkStep_SetOptions(
    ark_mem: &ARKodeMem,
    argidx: &mut i32,
    argv: &[String],
    offset: usize,
    arg_used: &mut sunbooleantype,
) -> i32 {
    /* Set lists of keys, and the corresponding set routines */
    static char_pairs: [sunKeyCharPair; 2] = [
        sunKeyCharPair { key: "sts_method_name", set: cliLSRKStepSetSTSMethodByName },
        sunKeyCharPair { key: "ssp_method_name", set: cliLSRKStepSetSSPMethodByName },
    ];
    let num_char_keys: i32 = char_pairs.len() as i32;

    static long_pairs: [sunKeyLongPair; 1] =
        [sunKeyLongPair { key: "dom_eig_frequency", set: cliLSRKStepSetDomEigFrequency }];
    let num_long_keys: i32 = long_pairs.len() as i32;

    static int_pairs: [sunKeyIntPair; 5] = [
        sunKeyIntPair { key: "max_num_stages", set: cliLSRKStepSetMaxNumStages },
        sunKeyIntPair { key: "num_ssp_stages", set: cliLSRKStepSetNumSSPStages },
        sunKeyIntPair {
            key: "num_dom_eig_est_init_preprocess_iters",
            set: cliLSRKStepSetNumDomEigEstInitPreprocessIters,
        },
        sunKeyIntPair {
            key: "num_dom_eig_est_preprocess_iters",
            set: cliLSRKStepSetNumDomEigEstPreprocessIters,
        },
        sunKeyIntPair {
            key: "use_analytic_stability_region",
            set: cliLSRKStepSetUseAnalyticStabilityRegion,
        },
    ];
    let num_int_keys: i32 = int_pairs.len() as i32;

    static real_pairs: [sunKeyRealPair; 1] =
        [sunKeyRealPair { key: "dom_eig_safety_factor", set: cliLSRKStepSetDomEigSafetyFactor }];
    let num_real_keys: i32 = real_pairs.len() as i32;

    /* the CLI helpers receive C's `void* ark_mem` as a boxed handle clone */
    let mut mem: Option<Box<dyn Any>> = Some(Box::new(ark_mem.clone()));

    /* check all "char" keys */
    let mut j: i32 = 0;
    let retval = sunCheckAndSetCharArgs(&mut mem, argidx, argv, offset, &char_pairs,
                                        num_char_keys, arg_used, &mut j);
    if retval != ARK_SUCCESS {
        arkProcessError(
            Some(ark_mem),
            retval,
            line!() as i32,
            "lsrkStep_SetOptions",
            file!(),
            &format!("error setting key: {}", char_pairs[j as usize].key),
        );
        return retval;
    }
    if *arg_used {
        return ARK_SUCCESS;
    }

    /* check all "long int" keys */
    let retval = sunCheckAndSetLongArgs(&mut mem, argidx, argv, offset, &long_pairs,
                                        num_long_keys, arg_used, &mut j);
    if retval != ARK_SUCCESS {
        arkProcessError(
            Some(ark_mem),
            retval,
            line!() as i32,
            "lsrkStep_SetOptions",
            file!(),
            &format!("error setting key: {}", long_pairs[j as usize].key),
        );
        return retval;
    }
    if *arg_used {
        return ARK_SUCCESS;
    }

    /* check all "int" keys */
    let retval = sunCheckAndSetIntArgs(&mut mem, argidx, argv, offset, &int_pairs, num_int_keys,
                                       arg_used, &mut j);
    if retval != ARK_SUCCESS {
        arkProcessError(
            Some(ark_mem),
            retval,
            line!() as i32,
            "lsrkStep_SetOptions",
            file!(),
            &format!("error setting key: {}", int_pairs[j as usize].key),
        );
        return retval;
    }
    if *arg_used {
        return ARK_SUCCESS;
    }

    /* check all "real" keys */
    let retval = sunCheckAndSetRealArgs(&mut mem, argidx, argv, offset, &real_pairs,
                                        num_real_keys, arg_used, &mut j);
    if retval != ARK_SUCCESS {
        arkProcessError(
            Some(ark_mem),
            retval,
            line!() as i32,
            "lsrkStep_SetOptions",
            file!(),
            &format!("error setting key: {}", real_pairs[j as usize].key),
        );
        return retval;
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  lsrkStep_SetDefaults:

  Resets all LSRKStep optional inputs to their default values.
  Does not change problem-defining function pointers or
  user_data pointer.
  ---------------------------------------------------------------*/
pub fn lsrkStep_SetDefaults(ark_mem: &ARKodeMem) -> i32 {
    /* access ARKodeLSRKStepMem structure */
    let retval = lsrkStep_AccessStepMem(ark_mem, "lsrkStep_SetDefaults");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* Set default values for integrator optional inputs
       (overwrite some adaptivity params for LSRKStep use) */
    {
        let mut step_mem = lsrkStep_mem_mut(ark_mem);
        step_mem.req_stages = 0; /* no stages */

        /* Spectral info */
        step_mem.dom_eig_safety = DOM_EIG_SAFETY_DEFAULT;
        step_mem.dom_eig_freq = DOM_EIG_FREQ_DEFAULT;
        step_mem.rkc_damping = RKC_DAMPING_DEFAULT;
        step_mem.const_Jac = SUNFALSE;
        step_mem.num_init_warmups = DOM_EIG_NUM_INIT_WARMUPS_DEFAULT;
        step_mem.num_warmups = DOM_EIG_NUM_WARMUPS_DEFAULT;
        step_mem.use_ellipse = SUNTRUE;
    }

    /* Load the default SUNAdaptController */
    let retval = arkReplaceAdaptController(ark_mem, None, SUNTRUE);
    if retval != 0 {
        return retval;
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  lsrkStep_GetStageIndex:

  Returns the current stage index and number of stages
  ---------------------------------------------------------------*/
pub fn lsrkStep_GetStageIndex(ark_mem: &ARKodeMem, stage: &mut i32, max_stages: &mut i32) -> i32 {
    /* access ARKodeLSRKStepMem structure */
    let retval = lsrkStep_AccessStepMem(ark_mem, "lsrkStep_GetStageIndex");
    if retval != ARK_SUCCESS {
        return retval;
    }

    let step_mem = lsrkStep_mem_mut(ark_mem);
    *stage = step_mem.istage;
    *max_stages = step_mem.req_stages + if step_mem.is_SSP { 0 } else { 1 };

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  lsrkStep_PrintAllStats:

  Prints integrator statistics for STS methods
  ---------------------------------------------------------------*/
pub fn lsrkStep_PrintAllStats(
    ark_mem: &ARKodeMem,
    outfile: &SUNFile,
    fmt: SUNOutputFormat,
) -> i32 {
    /* access ARKodeLSRKStepMem structure */
    let retval = lsrkStep_AccessStepMem(ark_mem, "lsrkStep_PrintAllStats");
    if retval != ARK_SUCCESS {
        return retval;
    }

    let (
        nfe,
        is_SSP,
        req_stages,
        dom_eig_num_evals,
        has_DEE,
        nfeDQ,
        num_dee_iters,
        stage_max,
        stage_max_limit,
        spectral_radius_max,
        spectral_radius_min,
    ) = {
        let step_mem = lsrkStep_mem_mut(ark_mem);
        (
            step_mem.nfe,
            step_mem.is_SSP,
            step_mem.req_stages,
            step_mem.dom_eig_num_evals,
            step_mem.DEE.is_some(),
            step_mem.nfeDQ,
            step_mem.num_dee_iters,
            step_mem.stage_max,
            step_mem.stage_max_limit,
            step_mem.spectral_radius_max,
            step_mem.spectral_radius_min,
        )
    };

    sunfprintf_long(outfile, fmt, SUNFALSE, "RHS fn evals", nfe);
    if is_SSP {
        sunfprintf_long(outfile, fmt, SUNFALSE, "Number of stages used", req_stages as i64);
    } else {
        sunfprintf_long(
            outfile,
            fmt,
            SUNFALSE,
            "Number of dom_eig updates",
            dom_eig_num_evals,
        );
        if has_DEE {
            sunfprintf_long(outfile, fmt, SUNFALSE, "Number of fe calls for DEE", nfeDQ);
            sunfprintf_long(
                outfile,
                fmt,
                SUNFALSE,
                "Number of iterations for DEE",
                num_dee_iters,
            );
        }
        sunfprintf_long(
            outfile,
            fmt,
            SUNFALSE,
            "Max. num. of stages used",
            stage_max as i64,
        );
        sunfprintf_long(
            outfile,
            fmt,
            SUNFALSE,
            "Max. num. of stages allowed",
            stage_max_limit as i64,
        );
        sunfprintf_real(
            outfile,
            fmt,
            SUNFALSE,
            "Max. spectral radius",
            spectral_radius_max,
        );
        sunfprintf_real(
            outfile,
            fmt,
            SUNFALSE,
            "Min. spectral radius",
            spectral_radius_min,
        );
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  lsrkStep_WriteParameters:

  Outputs all solver parameters to the provided file pointer.
  ---------------------------------------------------------------*/
pub fn lsrkStep_WriteParameters(ark_mem: &ARKodeMem, fp: &SUNFile) -> i32 {
    /* access ARKodeLSRKStepMem structure */
    let retval = lsrkStep_AccessStepMem(ark_mem, "lsrkStep_WriteParameters");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* print integrator parameters to file. C's `default:` arm is
       unreachable: the Rust enum has exactly the five upstream values. */
    let LSRKmethod = lsrkStep_mem_mut(ark_mem).LSRKmethod;
    match LSRKmethod {
        ARKODE_LSRK_RKC_2 => fp.write_str("LSRKStep RKC time step module parameters:\n"),
        ARKODE_LSRK_RKL_2 => fp.write_str("LSRKStep RKL time step module parameters:\n"),
        ARKODE_LSRK_SSP_S_2 => fp.write_str("LSRKStep SSP(s,2) time step module parameters:\n"),
        ARKODE_LSRK_SSP_S_3 => fp.write_str("LSRKStep SSP(s,3) time step module parameters:\n"),
        ARKODE_LSRK_SSP_10_4 => fp.write_str("LSRKStep SSP(10,4) time step module parameters:\n"),
    }

    let (
        q,
        p,
        is_SSP,
        req_stages,
        stage_max_limit,
        has_DEE,
        nfeDQ,
        spectral_radius,
        dom_eig_safety,
        use_ellipse,
        rkc_damping,
        dom_eig_freq,
        num_init_warmups,
        num_warmups,
        const_Jac,
    ) = {
        let step_mem = lsrkStep_mem_mut(ark_mem);
        (
            step_mem.q,
            step_mem.p,
            step_mem.is_SSP,
            step_mem.req_stages,
            step_mem.stage_max_limit,
            step_mem.DEE.is_some(),
            step_mem.nfeDQ,
            step_mem.spectral_radius,
            step_mem.dom_eig_safety,
            step_mem.use_ellipse,
            step_mem.rkc_damping,
            step_mem.dom_eig_freq,
            step_mem.num_init_warmups,
            step_mem.num_warmups,
            step_mem.const_Jac,
        )
    };

    fp.write_str(&format!("  Method order {q}\n"));
    fp.write_str(&format!("  Embedding order {p}\n"));

    /* C's `default:` arm over `sunbooleantype` is unreachable. */
    if is_SSP {
        fp.write_str(&format!("  Number of stages used = {req_stages}\n"));
    } else {
        fp.write_str(&format!(
            "  Maximum number of stages allowed = {stage_max_limit}\n"
        ));
        if has_DEE {
            fp.write_str(&format!("  Number of fe calls for DEE = {nfeDQ}\n"));
        }
        fp.write_str(&format!(
            "  Current spectral radius = {}\n",
            sun_format_g(spectral_radius)
        ));
        fp.write_str(&format!(
            "  Safety factor for the dom eig = {}\n",
            sun_format_g(dom_eig_safety)
        ));
        fp.write_str(&format!(
            "  Use elliptical stability region = {}\n",
            use_ellipse as i32
        ));
        fp.write_str(&format!(
            "  Damping factor for RKC = {}\n",
            sun_format_g(rkc_damping)
        ));
        fp.write_str(&format!(
            "  Max num of successful steps before new dom eig update = {dom_eig_freq}\n"
        ));
        fp.write_str(&format!(
            "  Number of first preprocessing warmups = {num_init_warmups}\n"
        ));
        fp.write_str(&format!(
            "  Number of subsequent preprocessing warmups = {num_warmups}\n"
        ));
        fp.write_str(&format!(
            "  Flag to indicate Jacobian is constant = {}\n",
            const_Jac as i32
        ));
    }

    fp.write_str("\n");

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  lsrkStep_GetNumRhsEvals:

  Returns the current number of RHS calls
  ---------------------------------------------------------------*/
pub fn lsrkStep_GetNumRhsEvals(
    ark_mem: &ARKodeMem,
    partition_index: i32,
    rhs_evals: &mut i64,
) -> i32 {
    /* access ARKodeLSRKStepMem structure */
    let retval = lsrkStep_AccessStepMem(ark_mem, "lsrkStep_GetNumRhsEvals");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* C checks `rhs_evals == NULL`; impossible through `&mut i64` */

    if partition_index > 0 {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "lsrkStep_GetNumRhsEvals",
            file!(),
            "Invalid partition index",
        );
        return ARK_ILL_INPUT;
    }

    *rhs_evals = lsrkStep_mem_mut(ark_mem).nfe;

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  lsrkStep_GetEstLocalErrors: Returns the current local truncation
  error estimate vector
  ---------------------------------------------------------------*/
pub fn lsrkStep_GetEstLocalErrors(ark_mem: &ARKodeMem, ele: &N_Vector) -> i32 {
    let retval = lsrkStep_AccessStepMem(ark_mem, "lsrkStep_GetEstLocalErrors");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* return an error if local truncation error is not computed */
    if ark_mem.borrow().fixedstep {
        return ARK_STEPPER_UNSUPPORTED;
    }

    /* otherwise, copy local truncation error vector to output */
    let tempv1 = ark_mem.borrow().tempv1.clone().expect("tempv1");
    N_VScale(ONE, &tempv1, ele);
    ARK_SUCCESS
}

/*===============================================================
  EOF
  ===============================================================*/
