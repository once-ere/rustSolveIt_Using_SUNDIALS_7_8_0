//! Port of `src/arkode/arkode_arkstep_io.c`: the optional input and
//! output functions for the ARKODE ARKStep time stepper module.
//!
//! Structure mirrors the C file exactly:
//!   1. exported optional *input* functions (`ARKStepSet*`),
//!   2. exported optional *output* functions,
//!   3. private functions attached to ARKODE (the `step_*` table entries),
//!   4. exported-but-deprecated user-callable wrappers that simply
//!      forward to the corresponding `ARKode*` routine.
//!
//! Conventions (workspace-wide): `void* arkode_mem` -> `&ARKodeMem`;
//! `T* out` -> `&mut T` in the same position with the same name;
//! `N_Vector*`/`SUNMatrix*` out-params -> `&mut Option<...>`;
//! nullable object arguments -> `Option<&...>`; `FILE*` -> `&SUNFile`;
//! `char*`-returning routines -> `String`. Every float printed goes
//! through `sundials_utils` (`SUN_FORMAT_G` -> `sun_format_g`).
//!
//! Borrow discipline: the `arkStep_mem_mut` guard *is* a borrow of the
//! `ARKodeMem`; it is never held across `arkProcessError`, a nested
//! `ARKStep*`/`ARKode*` call, an `N_Vector` operation, or a
//! (non)linear-solver call. Values are copied out in a scoped block
//! first.

use std::any::Any;

use sundials_core::sunadaptcontroller_soderlind::{
    SUNAdaptController_I, SUNAdaptController_PI, SUNAdaptController_PID,
    SUNAdaptController_SetParams_PI, SUNAdaptController_SetParams_PID,
};
use sundials_core::sundials_adaptcontroller::{
    SUNAdaptController, SUNAdaptController_Destroy, SUNAdaptController_SetErrorBias,
    SUNAdaptController_Space,
};
use sundials_core::sundials_cli::{
    sunCheckAndSetActionArgs, sunCheckAndSetTwoCharArgs, sunKeyActionPair, sunKeyTwoCharPair,
};
use sundials_core::sundials_errors::SUN_SUCCESS;
use sundials_core::sundials_linearsolver::SUNLinearSolver;
use sundials_core::sundials_matrix::SUNMatrix;
use sundials_core::sundials_nonlinearsolver::{
    SUNNonlinSolFree, SUNNonlinSolSetMaxIters, SUNNonlinearSolver,
};
use sundials_core::sundials_nvector::{N_VScale, N_Vector};
use sundials_core::sundials_types::*;
use sundials_core::sundials_utils::{sun_format_g, sunfprintf_long, sunfprintf_real, SUNFile};

use crate::arkode::{
    arkAllocVec, arkFreeVec, ARKodeCreateMRIStepInnerStepper,
    ARKodeEvolve, ARKodeFree, ARKodeGetDky, ARKodePrintMem, ARKodeReset, ARKodeResFtolerance,
    ARKodeResStolerance, ARKodeResVtolerance, ARKodeResize, ARKodeSStolerances,
    ARKodeSVtolerances, ARKodeWFtolerances,
};
use crate::arkode_arkstep::{
    arkStep_AccessARKODEStepMem, arkStep_AccessStepMem, arkStep_GetOrder, arkStep_RelaxDeltaE,
    arkStep_mem_mut, MSG_ARK_MISSING_F, MSG_ARK_MISSING_FE, MSG_ARK_MISSING_FI,
};
use crate::arkode_arkstep_nls::arkStep_SetNlsSysFn;
use crate::arkode_io::arkReplaceAdaptController;
use crate::arkode_butcher::{
    ARKodeButcherTable, ARKodeButcherTable_Copy, ARKodeButcherTable_Space,
    ARKodeButcherTable_Write,
};
use crate::arkode_butcher_dirk::{
    arkButcherTableDIRKNameToID, ARKodeButcherTable_LoadDIRK, ARKODE_ARK2_DIRK_3_1_2,
    ARKODE_ARK324L2SA_DIRK_4_2_3, ARKODE_ARK436L2SA_DIRK_6_3_4, ARKODE_ARK437L2SA_DIRK_7_3_4,
    ARKODE_ARK548L2SAb_DIRK_8_4_5, ARKODE_ARK548L2SA_DIRK_8_4_5, ARKODE_ASCHER_SDIRK_3_1_2,
    ARKODE_DIRKTableID, ARKODE_ESDIRK_4_2_3, ARKODE_MAX_DIRK_NUM, ARKODE_MIN_DIRK_NUM,
    ARKODE_SSP_DIRK_3_1_2, ARKODE_SSP_LSPUM_SDIRK_3_1_2,
};
use crate::arkode_butcher_erk::{
    arkButcherTableERKNameToID, ARKodeButcherTable_LoadERK, ARKODE_ARK2_ERK_3_1_2,
    ARKODE_ARK324L2SA_ERK_4_2_3, ARKODE_ARK436L2SA_ERK_6_3_4, ARKODE_ARK437L2SA_ERK_7_3_4,
    ARKODE_ARK548L2SAb_ERK_8_4_5, ARKODE_ARK548L2SA_ERK_8_4_5, ARKODE_ASCHER_ERK_3_1_2,
    ARKODE_ERKTableID, ARKODE_MAX_ERK_NUM, ARKODE_MIN_ERK_NUM, ARKODE_SSP_ERK_3_1_2,
    ARKODE_SSP_ERK_4_2_3, ARKODE_SSP_LSPUM_ERK_3_1_2,
};
use crate::arkode_impl::*;
use crate::arkode_io::{
    arkSetAdaptivityFn, arkSetAdaptivityMethod, ARKodeClearStopTime, ARKodeComputeState,
    ARKodeGetActualInitStep, ARKodeGetCurrentGamma, ARKodeGetCurrentState, ARKodeGetCurrentStep,
    ARKodeGetCurrentTime, ARKodeGetErrWeights, ARKodeGetEstLocalErrors, ARKodeGetLastStep,
    ARKodeGetNonlinSolvStats, ARKodeGetNonlinearSystemData, ARKodeGetNumAccSteps,
    ARKodeGetNumConstrFails, ARKodeGetNumErrTestFails, ARKodeGetNumExpSteps, ARKodeGetNumGEvals,
    ARKodeGetNumLinSolvSetups, ARKodeGetNumNonlinSolvConvFails, ARKodeGetNumNonlinSolvIters,
    ARKodeGetNumRhsEvals, ARKodeGetNumStepAttempts, ARKodeGetNumStepSolveFails, ARKodeGetNumSteps,
    ARKodeGetResWeights, ARKodeGetRootInfo, ARKodeGetReturnFlagName, ARKodeGetStepStats,
    ARKodeGetTolScaleFactor, ARKodeGetUserData, ARKodeGetWorkSpace, ARKodePrintAllStats,
    ARKodeSetAdaptController, ARKodeSetAdaptivityAdjustment, ARKodeSetCFLFraction,
    ARKodeSetConstraints, ARKodeSetDeduceImplicitRhs, ARKodeSetDefaults, ARKodeSetDeltaGammaMax,
    ARKodeSetErrorBias, ARKodeSetFixedStep, ARKodeSetFixedStepBounds, ARKodeSetInitStep,
    ARKodeSetInterpolantDegree, ARKodeSetInterpolantType, ARKodeSetInterpolateStopTime,
    ARKodeSetLSetupFrequency, ARKodeSetLinear, ARKodeSetMaxCFailGrowth, ARKodeSetMaxConvFails,
    ARKodeSetMaxEFailGrowth, ARKodeSetMaxErrTestFails, ARKodeSetMaxFirstGrowth, ARKodeSetMaxGrowth,
    ARKodeSetMaxHnilWarns, ARKodeSetMaxNonlinIters, ARKodeSetMaxNumConstrFails,
    ARKodeSetMaxNumSteps, ARKodeSetMaxStep, ARKodeSetMinReduction, ARKodeSetMinStep,
    ARKodeSetNlsRhsFn, ARKodeSetNoInactiveRootWarn, ARKodeSetNonlinConvCoef, ARKodeSetNonlinCRDown,
    ARKodeSetNonlinRDiv, ARKodeSetNonlinear, ARKodeSetNonlinearSolver, ARKodeSetOrder,
    ARKodeSetPostprocessStageFn, ARKodeSetPostprocessStepFn, ARKodeSetPredictorMethod,
    ARKodeSetRootDirection, ARKodeSetSafetyFactor, ARKodeSetSmallNumEFails,
    ARKodeSetStagePredictFn, ARKodeSetStabilityFn, ARKodeSetStopTime, ARKodeSetUserData,
    ARKodeWriteParameters,
};
use crate::arkode_ls::{
    arkls_mass_mem_mut, arkls_mem_mut, arkLSSetMassUserData, arkLSSetUserData,
    ARKodeGetCurrentMassMatrix, ARKodeGetJac, ARKodeGetJacNumSteps, ARKodeGetJacTime,
    ARKodeGetLastLinFlag, ARKodeGetLastMassFlag, ARKodeGetLinReturnFlagName,
    ARKodeGetLinWorkSpace, ARKodeGetMassWorkSpace, ARKodeGetNumJTSetupEvals, ARKodeGetNumJacEvals,
    ARKodeGetNumJtimesEvals, ARKodeGetNumLinConvFails, ARKodeGetNumLinIters,
    ARKodeGetNumLinRhsEvals, ARKodeGetNumMTSetups, ARKodeGetNumMassConvFails,
    ARKodeGetNumMassIters, ARKodeGetNumMassMult, ARKodeGetNumMassMultSetups,
    ARKodeGetNumMassPrecEvals, ARKodeGetNumMassPrecSolves, ARKodeGetNumMassSetups,
    ARKodeGetNumMassSolves, ARKodeGetNumPrecEvals, ARKodeGetNumPrecSolves, ARKodeSetEpsLin,
    ARKodeSetJacEvalFrequency, ARKodeSetJacFn, ARKodeSetJacTimes, ARKodeSetJacTimesRhsFn,
    ARKodeSetLSNormFactor, ARKodeSetLinSysFn, ARKodeSetLinearSolutionScaling,
    ARKodeSetLinearSolver, ARKodeSetMassEpsLin, ARKodeSetMassFn, ARKodeSetMassLSNormFactor,
    ARKodeSetMassLinearSolver, ARKodeSetMassPreconditioner, ARKodeSetMassTimes,
    ARKodeSetPreconditioner, ARKLsJacFn, ARKLsJacTimesSetupFn, ARKLsJacTimesVecFn, ARKLsLinSysFn,
    ARKLsMassFn, ARKLsMassPrecSetupFn, ARKLsMassPrecSolveFn, ARKLsMassTimesSetupFn,
    ARKLsMassTimesVecFn, ARKLsPrecSetupFn, ARKLsPrecSolveFn, ARKLS_SUCCESS,
};
use crate::arkode_mristep::MRIStepInnerStepper;
use crate::arkode_relaxation::{
    arkRelaxCreate, ARKodeGetNumRelaxBoundFails, ARKodeGetNumRelaxFails, ARKodeGetNumRelaxFnEvals,
    ARKodeGetNumRelaxJacEvals, ARKodeGetNumRelaxSolveFails, ARKodeGetNumRelaxSolveIters,
    ARKodeSetRelaxEtaFail, ARKodeSetRelaxFn, ARKodeSetRelaxLowerBound, ARKodeSetRelaxMaxFails,
    ARKodeSetRelaxMaxIters, ARKodeSetRelaxResTol, ARKodeSetRelaxSolver, ARKodeSetRelaxTol,
    ARKodeSetRelaxUpperBound,
};
use crate::arkode_root::ARKodeRootInit;

/*===============================================================
  Exported optional input functions.
  ===============================================================*/

/*---------------------------------------------------------------
  ARKStepSetExplicit:

  Specifies that the implicit portion of the problem is disabled,
  and to use an explicit RK method.
  ---------------------------------------------------------------*/
pub fn ARKStepSetExplicit(arkode_mem: &ARKodeMem) -> i32 {
    /* access ARKodeMem and ARKodeARKStepMem structures */
    let ark_mem = arkode_mem;
    let retval = arkStep_AccessARKODEStepMem(arkode_mem, "ARKStepSetExplicit");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* ensure that fe is defined */
    let fe = arkStep_mem_mut(ark_mem).fe;
    if fe.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "ARKStepSetExplicit",
            file!(),
            MSG_ARK_MISSING_FE,
        );
        return ARK_ILL_INPUT;
    }

    /* set the relevant parameters */
    {
        let mut step_mem = arkStep_mem_mut(ark_mem);
        step_mem.explicit = SUNTRUE;
        step_mem.implicit = SUNFALSE;
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ARKStepSetImplicit:

  Specifies that the explicit portion of the problem is disabled,
  and to use an implicit RK method.
  ---------------------------------------------------------------*/
pub fn ARKStepSetImplicit(arkode_mem: &ARKodeMem) -> i32 {
    /* access ARKodeMem and ARKodeARKStepMem structures */
    let ark_mem = arkode_mem;
    let mut retval = arkStep_AccessARKODEStepMem(arkode_mem, "ARKStepSetImplicit");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* ensure that fi is defined */
    let fi = arkStep_mem_mut(ark_mem).fi;
    if fi.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "ARKStepSetImplicit",
            file!(),
            MSG_ARK_MISSING_FI,
        );
        return ARK_ILL_INPUT;
    }

    /* set the relevant parameters */
    {
        let mut step_mem = arkStep_mem_mut(ark_mem);
        step_mem.implicit = SUNTRUE;
        step_mem.explicit = SUNFALSE;
    }

    /* re-attach internal error weight functions if necessary */
    let user_efun = ark_mem.borrow().user_efun;
    if !user_efun {
        let (itol, vabstol, reltol, sabstol) = {
            let m = ark_mem.borrow();
            (m.itol, m.Vabstol.clone(), m.reltol, m.Sabstol)
        };
        if itol == ARK_SV && vabstol.is_some() {
            retval = ARKodeSVtolerances(ark_mem, reltol, vabstol.as_ref().expect("Vabstol"));
        } else {
            retval = ARKodeSStolerances(ark_mem, reltol, sabstol);
        }
        if retval != ARK_SUCCESS {
            return retval;
        }
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ARKStepSetImEx:

  Specifies that the specifies that problem has both implicit and
  explicit parts, and to use an ARK method (this is the default).
  ---------------------------------------------------------------*/
pub fn ARKStepSetImEx(arkode_mem: &ARKodeMem) -> i32 {
    /* access ARKodeMem and ARKodeARKStepMem structures */
    let ark_mem = arkode_mem;
    let mut retval = arkStep_AccessARKODEStepMem(arkode_mem, "ARKStepSetImEx");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* ensure that fe and fi are defined */
    let fe = arkStep_mem_mut(ark_mem).fe;
    if fe.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "ARKStepSetImEx",
            file!(),
            MSG_ARK_MISSING_FE,
        );
        return ARK_ILL_INPUT;
    }
    let fi = arkStep_mem_mut(ark_mem).fi;
    if fi.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "ARKStepSetImEx",
            file!(),
            MSG_ARK_MISSING_FI,
        );
        return ARK_ILL_INPUT;
    }

    /* set the relevant parameters */
    {
        let mut step_mem = arkStep_mem_mut(ark_mem);
        step_mem.explicit = SUNTRUE;
        step_mem.implicit = SUNTRUE;
    }

    /* re-attach internal error weight functions if necessary */
    let user_efun = ark_mem.borrow().user_efun;
    if !user_efun {
        let (itol, vabstol, reltol, sabstol) = {
            let m = ark_mem.borrow();
            (m.itol, m.Vabstol.clone(), m.reltol, m.Sabstol)
        };
        if itol == ARK_SV && vabstol.is_some() {
            retval = ARKodeSVtolerances(ark_mem, reltol, vabstol.as_ref().expect("Vabstol"));
        } else {
            retval = ARKodeSStolerances(ark_mem, reltol, sabstol);
        }
        if retval != ARK_SUCCESS {
            return retval;
        }
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ARKStepSetTables:

  Specifies to use customized Butcher tables for the system.

  If Bi is NULL, then this sets the integrator in 'explicit' mode.

  If Be is NULL, then this sets the integrator in 'implicit' mode.

  Returns ARK_ILL_INPUT if both Butcher tables are not supplied.
  ---------------------------------------------------------------*/
pub fn ARKStepSetTables(
    arkode_mem: &ARKodeMem,
    q: i32,
    p: i32,
    Bi: Option<&ARKodeButcherTable>,
    Be: Option<&ARKodeButcherTable>,
) -> i32 {
    let mut retval: i32;
    let ark_mem = arkode_mem;
    let mut Blrw: sunindextype = 0;
    let mut Bliw: sunindextype = 0;

    /* access ARKodeMem and ARKodeARKStepMem structures */
    retval = arkStep_AccessARKODEStepMem(arkode_mem, "ARKStepSetTables");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* check for illegal inputs */
    if Bi.is_none() && Be.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "ARKStepSetTables",
            file!(),
            "At least one complete table must be supplied",
        );
        return ARK_ILL_INPUT;
    }

    /* if both tables are set, check that they have the same number of stages */
    if Bi.is_some() && Be.is_some() {
        let bi_stages = Bi.expect("Bi").borrow().stages;
        let be_stages = Be.expect("Be").borrow().stages;
        if bi_stages != be_stages {
            arkProcessError(
                Some(ark_mem),
                ARK_MEM_NULL,
                line!() as i32,
                "ARKStepSetTables",
                file!(),
                "Both tables must have the same number of stages",
            );
            return ARK_ILL_INPUT;
        }
    }

    /* clear any existing parameters and Butcher tables */
    {
        let mut step_mem = arkStep_mem_mut(ark_mem);
        step_mem.stages = 0;
        step_mem.q = 0;
        step_mem.p = 0;
    }

    let old_be = { arkStep_mem_mut(ark_mem).Be.take() };
    ARKodeButcherTable_Space(old_be.as_ref(), &mut Bliw, &mut Blrw);
    drop(old_be); /* ARKodeButcherTable_Free(step_mem->Be); step_mem->Be = NULL */
    {
        let mut m = ark_mem.borrow_mut();
        m.liw -= Bliw;
        m.lrw -= Blrw;
    }

    let old_bi = { arkStep_mem_mut(ark_mem).Bi.take() };
    ARKodeButcherTable_Space(old_bi.as_ref(), &mut Bliw, &mut Blrw);
    drop(old_bi); /* ARKodeButcherTable_Free(step_mem->Bi); step_mem->Bi = NULL */
    {
        let mut m = ark_mem.borrow_mut();
        m.liw -= Bliw;
        m.lrw -= Blrw;
    }

    /*
     * determine mode (implicit/explicit/ImEx), and perform appropriate actions
     */

    /* explicit */
    if Bi.is_none() {
        let be = Be.expect("Be");

        /* set the relevant parameters (use table q and p) */
        {
            let (be_stages, be_q, be_p) = {
                let b = be.borrow();
                (b.stages, b.q, b.p)
            };
            let mut step_mem = arkStep_mem_mut(ark_mem);
            step_mem.stages = be_stages;
            step_mem.q = be_q;
            step_mem.p = be_p;
        }

        /* copy the table in step memory */
        let copy = ARKodeButcherTable_Copy(Some(&be));
        let copied = copy.is_some();
        arkStep_mem_mut(ark_mem).Be = copy;
        if !copied {
            arkProcessError(
                Some(ark_mem),
                ARK_MEM_NULL,
                line!() as i32,
                "ARKStepSetTables",
                file!(),
                MSG_ARK_NO_MEM,
            );
            return ARK_MEM_NULL;
        }

        /* set method as purely explicit */
        retval = ARKStepSetExplicit(arkode_mem);
        if retval != ARK_SUCCESS {
            arkProcessError(
                Some(ark_mem),
                ARK_ILL_INPUT,
                line!() as i32,
                "ARKStepSetTables",
                file!(),
                "Error in ARKStepSetExplicit",
            );
            return retval;
        }

        /* implicit */
    } else if Be.is_none() {
        let bi = Bi.expect("Bi");

        /* set the relevant parameters (use table q and p) */
        {
            let (bi_stages, bi_q, bi_p) = {
                let b = bi.borrow();
                (b.stages, b.q, b.p)
            };
            let mut step_mem = arkStep_mem_mut(ark_mem);
            step_mem.stages = bi_stages;
            step_mem.q = bi_q;
            step_mem.p = bi_p;
        }

        /* copy the table in step memory */
        let copy = ARKodeButcherTable_Copy(Some(&bi));
        let copied = copy.is_some();
        arkStep_mem_mut(ark_mem).Bi = copy;
        if !copied {
            arkProcessError(
                Some(ark_mem),
                ARK_MEM_NULL,
                line!() as i32,
                "ARKStepSetTables",
                file!(),
                MSG_ARK_NO_MEM,
            );
            return ARK_MEM_NULL;
        }

        /* set method as purely implicit */
        retval = ARKStepSetImplicit(arkode_mem);
        if retval != ARK_SUCCESS {
            arkProcessError(
                Some(ark_mem),
                ARK_ILL_INPUT,
                line!() as i32,
                "ARKStepSetTables",
                file!(),
                "Error in ARKStepSetImplicit",
            );
            return ARK_ILL_INPUT;
        }

        /* ImEx */
    } else {
        let bi = Bi.expect("Bi");
        let be = Be.expect("Be");

        /* set the relevant parameters (use input q and p) */
        {
            let bi_stages = bi.borrow().stages;
            let mut step_mem = arkStep_mem_mut(ark_mem);
            step_mem.stages = bi_stages;
            step_mem.q = q;
            step_mem.p = p;
        }

        /* copy the explicit table into step memory */
        let copy = ARKodeButcherTable_Copy(Some(&be));
        let copied = copy.is_some();
        arkStep_mem_mut(ark_mem).Be = copy;
        if !copied {
            arkProcessError(
                Some(ark_mem),
                ARK_MEM_NULL,
                line!() as i32,
                "ARKStepSetTables",
                file!(),
                MSG_ARK_NO_MEM,
            );
            return ARK_MEM_NULL;
        }

        /* copy the implicit table into step memory */
        let copy = ARKodeButcherTable_Copy(Some(&bi));
        let copied = copy.is_some();
        arkStep_mem_mut(ark_mem).Bi = copy;
        if !copied {
            arkProcessError(
                Some(ark_mem),
                ARK_MEM_NULL,
                line!() as i32,
                "ARKStepSetTables",
                file!(),
                MSG_ARK_NO_MEM,
            );
            return ARK_MEM_NULL;
        }

        /* set method as ImEx */
        retval = ARKStepSetImEx(arkode_mem);
        if retval != ARK_SUCCESS {
            arkProcessError(
                Some(ark_mem),
                ARK_ILL_INPUT,
                line!() as i32,
                "ARKStepSetTables",
                file!(),
                "Error in ARKStepSetImEx",
            );
            return ARK_ILL_INPUT;
        }
    }

    /* note Butcher table space requirements */
    let cur_be = { arkStep_mem_mut(ark_mem).Be.clone() };
    ARKodeButcherTable_Space(cur_be.as_ref(), &mut Bliw, &mut Blrw);
    {
        let mut m = ark_mem.borrow_mut();
        m.liw += Bliw;
        m.lrw += Blrw;
    }

    let cur_bi = { arkStep_mem_mut(ark_mem).Bi.clone() };
    ARKodeButcherTable_Space(cur_bi.as_ref(), &mut Bliw, &mut Blrw);
    {
        let mut m = ark_mem.borrow_mut();
        m.liw += Bliw;
        m.lrw += Blrw;
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ARKStepSetTableNum:

  Specifies to use pre-existing Butcher tables for the system,
  based on the integer flags passed to
  ARKodeButcherTable_LoadERK() and ARKodeButcherTable_LoadDIRK()
  within the files arkode_butcher_erk.c and arkode_butcher_dirk.c
  (automatically calls ARKStepSetImEx).

  If either argument is negative (illegal), then this disables the
  corresponding table (e.g. itable = -1  ->  explicit)

  Note: this routine should NOT be used in conjunction with
  ARKodeSetOrder.
  ---------------------------------------------------------------*/
pub fn ARKStepSetTableNum(
    arkode_mem: &ARKodeMem,
    itable: ARKODE_DIRKTableID,
    etable: ARKODE_ERKTableID,
) -> i32 {
    let flag: i32;
    let retval: i32;
    let ark_mem = arkode_mem;
    let mut Blrw: sunindextype = 0;
    let mut Bliw: sunindextype = 0;

    /* access ARKodeMem and ARKodeARKStepMem structures */
    retval = arkStep_AccessARKODEStepMem(arkode_mem, "ARKStepSetTableNum");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* clear any existing parameters and Butcher tables */
    {
        let mut step_mem = arkStep_mem_mut(ark_mem);
        step_mem.stages = 0;
        step_mem.q = 0;
        step_mem.p = 0;
    }

    let old_be = { arkStep_mem_mut(ark_mem).Be.take() };
    ARKodeButcherTable_Space(old_be.as_ref(), &mut Bliw, &mut Blrw);
    drop(old_be); /* ARKodeButcherTable_Free(step_mem->Be); step_mem->Be = NULL */
    {
        let mut m = ark_mem.borrow_mut();
        m.liw -= Bliw;
        m.lrw -= Blrw;
    }

    let old_bi = { arkStep_mem_mut(ark_mem).Bi.take() };
    ARKodeButcherTable_Space(old_bi.as_ref(), &mut Bliw, &mut Blrw);
    drop(old_bi); /* ARKodeButcherTable_Free(step_mem->Bi); step_mem->Bi = NULL */
    {
        let mut m = ark_mem.borrow_mut();
        m.liw -= Bliw;
        m.lrw -= Blrw;
    }

    /* determine mode (implicit/explicit/ImEx), and perform
       appropriate actions  */

    /*     illegal inputs */
    if (itable < 0) && (etable < 0) {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "ARKStepSetTableNum",
            file!(),
            "At least one valid table number must be supplied",
        );
        return ARK_ILL_INPUT;

    /* explicit */
    } else if itable < 0 {
        /* check that argument specifies an explicit table */
        if etable < ARKODE_MIN_ERK_NUM || etable > ARKODE_MAX_ERK_NUM {
            arkProcessError(
                Some(ark_mem),
                ARK_MEM_NULL,
                line!() as i32,
                "ARKStepSetTableNum",
                file!(),
                "Illegal ERK table number",
            );
            return ARK_ILL_INPUT;
        }

        /* fill in table based on argument */
        let loaded = ARKodeButcherTable_LoadERK(etable);
        let ok = loaded.is_some();
        arkStep_mem_mut(ark_mem).Be = loaded;
        if !ok {
            arkProcessError(
                Some(ark_mem),
                ARK_MEM_NULL,
                line!() as i32,
                "ARKStepSetTableNum",
                file!(),
                "Error setting explicit table with that index",
            );
            return ARK_ILL_INPUT;
        }
        {
            let mut step_mem = arkStep_mem_mut(ark_mem);
            let (be_stages, be_q, be_p) = {
                let b = step_mem.Be.as_ref().expect("Be").borrow();
                (b.stages, b.q, b.p)
            };
            step_mem.stages = be_stages;
            step_mem.q = be_q;
            step_mem.p = be_p;
        }

        /* set method as purely explicit */
        flag = ARKStepSetExplicit(arkode_mem);
        if flag != ARK_SUCCESS {
            arkProcessError(
                Some(ark_mem),
                ARK_ILL_INPUT,
                line!() as i32,
                "ARKStepSetTableNum",
                file!(),
                "Error in ARKStepSetExplicit",
            );
            return flag;
        }

    /* implicit */
    } else if etable < 0 {
        /* check that argument specifies an implicit table */
        if itable < ARKODE_MIN_DIRK_NUM || itable > ARKODE_MAX_DIRK_NUM {
            arkProcessError(
                Some(ark_mem),
                ARK_MEM_NULL,
                line!() as i32,
                "ARKStepSetTableNum",
                file!(),
                "Illegal IRK table number",
            );
            return ARK_ILL_INPUT;
        }

        /* fill in table based on argument */
        let loaded = ARKodeButcherTable_LoadDIRK(itable);
        let ok = loaded.is_some();
        arkStep_mem_mut(ark_mem).Bi = loaded;
        if !ok {
            arkProcessError(
                Some(ark_mem),
                ARK_MEM_NULL,
                line!() as i32,
                "ARKStepSetTableNum",
                file!(),
                "Error setting table with that index",
            );
            return ARK_ILL_INPUT;
        }
        {
            let mut step_mem = arkStep_mem_mut(ark_mem);
            let (bi_stages, bi_q, bi_p) = {
                let b = step_mem.Bi.as_ref().expect("Bi").borrow();
                (b.stages, b.q, b.p)
            };
            step_mem.stages = bi_stages;
            step_mem.q = bi_q;
            step_mem.p = bi_p;
        }

        /* set method as purely implicit */
        flag = ARKStepSetImplicit(arkode_mem);
        if flag != ARK_SUCCESS {
            arkProcessError(
                Some(ark_mem),
                ARK_ILL_INPUT,
                line!() as i32,
                "ARKStepSetTableNum",
                file!(),
                "Error in ARKStepSetImplicit",
            );
            return flag;
        }

    /* ImEx */
    } else {
        /* ensure that tables match */
        if !((etable == ARKODE_ARK324L2SA_ERK_4_2_3) && (itable == ARKODE_ARK324L2SA_DIRK_4_2_3))
            && !((etable == ARKODE_ARK436L2SA_ERK_6_3_4)
                && (itable == ARKODE_ARK436L2SA_DIRK_6_3_4))
            && !((etable == ARKODE_ARK437L2SA_ERK_7_3_4)
                && (itable == ARKODE_ARK437L2SA_DIRK_7_3_4))
            && !((etable == ARKODE_ARK548L2SA_ERK_8_4_5)
                && (itable == ARKODE_ARK548L2SA_DIRK_8_4_5))
            && !((etable == ARKODE_ARK548L2SAb_ERK_8_4_5)
                && (itable == ARKODE_ARK548L2SAb_DIRK_8_4_5))
            && !((etable == ARKODE_ARK2_ERK_3_1_2) && (itable == ARKODE_ARK2_DIRK_3_1_2))
            && !((etable == ARKODE_ASCHER_ERK_3_1_2) && (itable == ARKODE_ASCHER_SDIRK_3_1_2))
            /*New Embedded IMEX-SSP Methods*/
            && !((etable == ARKODE_SSP_ERK_3_1_2) && (itable == ARKODE_SSP_DIRK_3_1_2))
            && !((etable == ARKODE_SSP_LSPUM_ERK_3_1_2)
                && (itable == ARKODE_SSP_LSPUM_SDIRK_3_1_2))
            && !((etable == ARKODE_SSP_ERK_4_2_3) && (itable == ARKODE_ESDIRK_4_2_3))
        {
            arkProcessError(
                Some(ark_mem),
                ARK_ILL_INPUT,
                line!() as i32,
                "ARKStepSetTableNum",
                file!(),
                "Incompatible Butcher tables for ARK method",
            );
            return ARK_ILL_INPUT;
        }

        /* fill in tables based on arguments */
        let loaded_bi = ARKodeButcherTable_LoadDIRK(itable);
        let loaded_be = ARKodeButcherTable_LoadERK(etable);
        let bi_ok = loaded_bi.is_some();
        let be_ok = loaded_be.is_some();
        {
            let mut step_mem = arkStep_mem_mut(ark_mem);
            step_mem.Bi = loaded_bi;
            step_mem.Be = loaded_be;
        }
        if !bi_ok {
            arkProcessError(
                Some(ark_mem),
                ARK_MEM_NULL,
                line!() as i32,
                "ARKStepSetTableNum",
                file!(),
                "Illegal IRK table number",
            );
            return ARK_ILL_INPUT;
        }
        if !be_ok {
            arkProcessError(
                Some(ark_mem),
                ARK_MEM_NULL,
                line!() as i32,
                "ARKStepSetTableNum",
                file!(),
                "Illegal ERK table number",
            );
            return ARK_ILL_INPUT;
        }
        {
            let mut step_mem = arkStep_mem_mut(ark_mem);
            let (bi_stages, bi_q, bi_p) = {
                let b = step_mem.Bi.as_ref().expect("Bi").borrow();
                (b.stages, b.q, b.p)
            };
            step_mem.stages = bi_stages;
            step_mem.q = bi_q;
            step_mem.p = bi_p;
        }

        /* set method as ImEx */
        if ARKStepSetImEx(arkode_mem) != ARK_SUCCESS {
            arkProcessError(
                Some(ark_mem),
                ARK_ILL_INPUT,
                line!() as i32,
                "ARKStepSetTableNum",
                file!(),
                MSG_ARK_MISSING_F,
            );
            return ARK_ILL_INPUT;
        }
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ARKStepSetTableName:

  Specifies to use pre-existing Butcher tables for the system,
  based on the string passed to
  ARKodeButcherTable_LoadERKByName() and
  ARKodeButcherTable_LoadDIRKByName() within the files
  arkode_butcher_erk.c and arkode_butcher_dirk.c (automatically
  calls ARKStepSetImEx).

  If itable is "ARKODE_DIRK_NONE" or etable is "ARKODE_ERK_NONE",
  then this disables the corresponding table.
  ---------------------------------------------------------------*/
pub fn ARKStepSetTableName(arkode_mem: &ARKodeMem, itable: &str, etable: &str) -> i32 {
    ARKStepSetTableNum(
        arkode_mem,
        arkButcherTableDIRKNameToID(itable),
        arkButcherTableERKNameToID(etable),
    )
}

/*===============================================================
  Exported optional output functions.
  ===============================================================*/

/*---------------------------------------------------------------
  arkStep_GetNumRhsEvals:

  Returns the current number of calls
  ---------------------------------------------------------------*/
pub fn arkStep_GetNumRhsEvals(
    ark_mem: &ARKodeMem,
    partition_index: i32,
    rhs_evals: &mut i64,
) -> i32 {
    /* access ARKodeARKStepMem structure */
    let retval = arkStep_AccessStepMem(ark_mem, "arkStep_GetNumRhsEvals");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* the C `rhs_evals == NULL` check is handled by the type system */

    if partition_index > 1 {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "arkStep_GetNumRhsEvals",
            file!(),
            "Invalid partition index",
        );
        return ARK_ILL_INPUT;
    }

    {
        let step_mem = arkStep_mem_mut(ark_mem);
        match partition_index {
            0 => *rhs_evals = step_mem.nfe,
            1 => *rhs_evals = step_mem.nfi,
            _ => *rhs_evals = step_mem.nfe + step_mem.nfi,
        }
    }

    ARK_SUCCESS
}

pub fn ARKStepGetNumRhsEvals(
    arkode_mem: &ARKodeMem,
    fe_evals: &mut i64,
    fi_evals: &mut i64,
) -> i32 {
    let mut retval;

    retval = ARKodeGetNumRhsEvals(arkode_mem, 0, fe_evals);
    if retval != ARK_SUCCESS {
        return retval;
    }

    retval = ARKodeGetNumRhsEvals(arkode_mem, 1, fi_evals);
    if retval != ARK_SUCCESS {
        return retval;
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ARKStepGetCurrentButcherTables:

  Sets pointers to the explicit and implicit Butcher tables
  currently in use.
  ---------------------------------------------------------------*/
pub fn ARKStepGetCurrentButcherTables(
    arkode_mem: &ARKodeMem,
    Bi: &mut Option<ARKodeButcherTable>,
    Be: &mut Option<ARKodeButcherTable>,
) -> i32 {
    /* access ARKodeMem and ARKodeARKStepMem structures */
    let ark_mem = arkode_mem;
    let retval = arkStep_AccessARKODEStepMem(arkode_mem, "ARKStepGetCurrentButcherTables");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* get tables from step_mem */
    {
        let step_mem = arkStep_mem_mut(ark_mem);
        *Bi = step_mem.Bi.clone();
        *Be = step_mem.Be.clone();
    }
    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ARKStepGetTimestepperStats:

  Returns integrator statistics
  ---------------------------------------------------------------*/
#[allow(clippy::too_many_arguments)]
pub fn ARKStepGetTimestepperStats(
    arkode_mem: &ARKodeMem,
    expsteps: &mut i64,
    accsteps: &mut i64,
    step_attempts: &mut i64,
    fe_evals: &mut i64,
    fi_evals: &mut i64,
    nlinsetups: &mut i64,
    netfails: &mut i64,
) -> i32 {
    /* access ARKodeMem and ARKodeARKStepMem structures */
    let ark_mem = arkode_mem;
    let retval = arkStep_AccessARKODEStepMem(arkode_mem, "ARKStepGetTimestepperStats");
    if retval != ARK_SUCCESS {
        return retval;
    }

    {
        let m = ark_mem.borrow();
        let hadapt_mem = m.hadapt_mem.as_ref().expect("hadapt_mem");

        /* set expsteps and accsteps from adaptivity structure */
        *expsteps = hadapt_mem.nst_exp;
        *accsteps = hadapt_mem.nst_acc;

        /* set remaining outputs */
        *step_attempts = m.nst_attempts;
        *netfails = m.netf;
    }
    {
        let step_mem = arkStep_mem_mut(ark_mem);
        *fe_evals = step_mem.nfe;
        *fi_evals = step_mem.nfi;
        *nlinsetups = step_mem.nsetups;
    }

    ARK_SUCCESS
}

/*===============================================================
  Private functions attached to ARKODE
  ===============================================================*/

/* -----------------------------------------------------------------
 * CLI adapter helpers: the C key tables hold pointers to the public
 * `ARKStepSet*` routines, which receive the raw `void* ark_mem`
 * forwarded by the `sunCheckAndSet*Args` helpers. Here each table entry
 * is a small adapter matching `sundials_core::sundials_cli`'s setter fn
 * types: it downcasts the token (an `Option<Box<dyn Any>>` holding an
 * `ARKodeMem` clone) back to the handle and forwards to the real
 * setter. A missing/mistyped token corresponds to C passing a garbage
 * pointer (UB) and maps to a deterministic panic.
 * ----------------------------------------------------------------- */

fn cliARKodeMem(mem: &mut Option<Box<dyn Any>>) -> ARKodeMem {
    mem.as_mut()
        .and_then(|b| b.downcast_ref::<ARKodeMem>())
        .cloned()
        .expect("ark_mem token")
}

fn cliARKStepSetTableName(mem: &mut Option<Box<dyn Any>>, arg1: &str, arg2: &str) -> i32 {
    let ark_mem = cliARKodeMem(mem);
    ARKStepSetTableName(&ark_mem, arg1, arg2)
}

fn cliARKStepSetExplicit(mem: &mut Option<Box<dyn Any>>) -> i32 {
    let ark_mem = cliARKodeMem(mem);
    ARKStepSetExplicit(&ark_mem)
}

fn cliARKStepSetImplicit(mem: &mut Option<Box<dyn Any>>) -> i32 {
    let ark_mem = cliARKodeMem(mem);
    ARKStepSetImplicit(&ark_mem)
}

fn cliARKStepSetImEx(mem: &mut Option<Box<dyn Any>>) -> i32 {
    let ark_mem = cliARKodeMem(mem);
    ARKStepSetImEx(&ark_mem)
}

/*---------------------------------------------------------------
  arkStep_SetOption:

  Provides string-based control over ARKStep-specific "set" routines.
  ---------------------------------------------------------------*/
pub fn arkStep_SetOptions(
    ark_mem: &ARKodeMem,
    argidx: &mut i32,
    argv: &[String],
    offset: usize,
    arg_used: &mut sunbooleantype,
) -> i32 {
    /* Set lists of keys, and the corresponding set routines */
    static twochar_pairs: [sunKeyTwoCharPair; 1] = [sunKeyTwoCharPair {
        key: "table_names",
        set: cliARKStepSetTableName,
    }];
    let num_twochar_keys: i32 = twochar_pairs.len() as i32;

    static action_pairs: [sunKeyActionPair; 3] = [
        sunKeyActionPair {
            key: "explicit",
            set: cliARKStepSetExplicit,
        },
        sunKeyActionPair {
            key: "implicit",
            set: cliARKStepSetImplicit,
        },
        sunKeyActionPair {
            key: "imex",
            set: cliARKStepSetImEx,
        },
    ];
    let num_action_keys: i32 = action_pairs.len() as i32;

    /* check all "twochar" keys */
    let mut j: i32 = 0;
    let mut retval: i32;
    let mut mem: Option<Box<dyn Any>> = Some(Box::new(ark_mem.clone()));
    retval = sunCheckAndSetTwoCharArgs(
        &mut mem,
        argidx,
        argv,
        offset,
        &twochar_pairs,
        num_twochar_keys,
        arg_used,
        &mut j,
    );
    if retval != ARK_SUCCESS {
        arkProcessError(
            Some(ark_mem),
            retval,
            line!() as i32,
            "arkStep_SetOptions",
            file!(),
            &format!(
                "error setting command-line argument: {}",
                twochar_pairs[j as usize].key
            ),
        );
        return retval;
    }
    if *arg_used {
        return ARK_SUCCESS;
    }

    /* check all action keys */
    retval = sunCheckAndSetActionArgs(
        &mut mem,
        argidx,
        argv,
        offset,
        &action_pairs,
        num_action_keys,
        arg_used,
        &mut j,
    );
    if retval != ARK_SUCCESS {
        arkProcessError(
            Some(ark_mem),
            retval,
            line!() as i32,
            "arkStep_SetOptions",
            file!(),
            &format!(
                "error setting command-line argument: {}",
                action_pairs[j as usize].key
            ),
        );
        return retval;
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  arkStep_SetRelaxFn:

  Sets up the relaxation module using ARKStep's utility routines.
  ---------------------------------------------------------------*/
pub fn arkStep_SetRelaxFn(
    ark_mem: &ARKodeMem,
    rfn: Option<ARKRelaxFn>,
    rjac: Option<ARKRelaxJacFn>,
) -> i32 {
    arkRelaxCreate(
        ark_mem,
        rfn,
        rjac,
        Some(arkStep_RelaxDeltaE),
        Some(arkStep_GetOrder),
    )
}

/*---------------------------------------------------------------
  arkStep_SetUserData:

  Passes user-data pointer to attached linear solver modules.
  ---------------------------------------------------------------*/
pub fn arkStep_SetUserData(ark_mem: &ARKodeMem, user_data: &mut Option<Box<dyn Any>>) -> i32 {
    let mut retval;

    /* access ARKodeARKStepMem structure */
    retval = arkStep_AccessStepMem(ark_mem, "arkStep_SetUserData");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* set user data in ARKODE LS mem (C: step_mem->lmem != NULL; the ARKLS
    record is stored by value in ark_mem.ark_lmem -- see arkode_impl.rs) */
    if ark_mem.borrow().ark_lmem.is_some() {
        retval = arkLSSetUserData(ark_mem, user_data);
        if retval != ARKLS_SUCCESS {
            return retval;
        }
    }

    /* set user data in ARKODE LSMass mem */
    if ark_mem.borrow().ark_mass_mem.is_some() {
        retval = arkLSSetMassUserData(ark_mem, user_data);
        if retval != ARKLS_SUCCESS {
            return retval;
        }
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  arkStep_SetDefaults:

  Resets all ARKStep optional inputs to their default values.
  Does not change problem-defining function pointers or
  user_data pointer.  Also leaves alone any data
  structures/options related to the ARKODE infrastructure itself
  (e.g., root-finding and post-process step).
  ---------------------------------------------------------------*/
pub fn arkStep_SetDefaults(ark_mem: &ARKodeMem) -> i32 {
    let mut Blrw: sunindextype = 0;
    let mut Bliw: sunindextype = 0;
    let retval;

    /* access ARKodeARKStepMem structure */
    let access = arkStep_AccessStepMem(ark_mem, "arkStep_SetDefaults");
    if access != ARK_SUCCESS {
        return access;
    }

    /* Set default values for integrator optional inputs */
    {
        let mut step_mem = arkStep_mem_mut(ark_mem);
        step_mem.q = Q_DEFAULT; /* method order */
        step_mem.p = 0; /* embedding order */
        step_mem.predictor = 0; /* trivial predictor */
        step_mem.linear = SUNFALSE; /* nonlinear problem */
        step_mem.linear_timedep = SUNTRUE; /* dfi/dy depends on t */
        step_mem.autonomous = SUNFALSE; /* non-autonomous problem */
        step_mem.explicit = SUNTRUE; /* fe(t,y) will be used */
        step_mem.implicit = SUNTRUE; /* fi(t,y) will be used */
        step_mem.deduce_rhs = SUNFALSE; /* deduce fi on result of NLS */
        step_mem.maxcor = MAXCOR; /* max nonlinear iters/stage */
        step_mem.nlscoef = NLSCOEF; /* nonlinear tolerance coefficient */
        step_mem.crdown = CRDOWN; /* nonlinear convergence estimate coeff. */
        step_mem.rdiv = RDIV; /* nonlinear divergence tolerance */
        step_mem.dgmax = DGMAX; /* max step change before recomputing J or P */
        step_mem.msbp = MSBP; /* max steps between updates to J or P */
        step_mem.stages = 0; /* no stages */
        step_mem.istage = 0; /* current stage */
        step_mem.jcur.set(SUNFALSE);
        step_mem.convfail = ARK_NO_FAILURES;
        step_mem.stage_predict = None; /* no user-supplied stage predictor */
    }

    /* Remove pre-existing Butcher tables */
    let old_be = { arkStep_mem_mut(ark_mem).Be.take() };
    if old_be.is_some() {
        ARKodeButcherTable_Space(old_be.as_ref(), &mut Bliw, &mut Blrw);
        {
            let mut m = ark_mem.borrow_mut();
            m.liw -= Bliw;
            m.lrw -= Blrw;
        }
        drop(old_be); /* ARKodeButcherTable_Free(step_mem->Be) */
    }
    /* step_mem->Be = NULL (already taken above) */

    let old_bi = { arkStep_mem_mut(ark_mem).Bi.take() };
    if old_bi.is_some() {
        ARKodeButcherTable_Space(old_bi.as_ref(), &mut Bliw, &mut Blrw);
        {
            let mut m = ark_mem.borrow_mut();
            m.liw -= Bliw;
            m.lrw -= Blrw;
        }
        drop(old_bi); /* ARKodeButcherTable_Free(step_mem->Bi) */
    }
    /* step_mem->Bi = NULL (already taken above) */

    /* Remove pre-existing nonlinear solver object */
    let (old_nls, own_nls) = {
        let mut step_mem = arkStep_mem_mut(ark_mem);
        (step_mem.NLS.take(), step_mem.ownNLS)
    };
    if old_nls.is_some() && own_nls {
        let _ = SUNNonlinSolFree(old_nls);
    }
    /* step_mem->NLS = NULL (already taken above) */

    /* Load the default SUNAdaptController */
    retval = arkReplaceAdaptController(ark_mem, None, SUNTRUE);
    if retval != 0 {
        return retval;
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  arkStep_SetOrder:

  Specifies the method order
  ---------------------------------------------------------------*/
pub fn arkStep_SetOrder(ark_mem: &ARKodeMem, ord: i32) -> i32 {
    let mut Blrw: sunindextype = 0;
    let mut Bliw: sunindextype = 0;

    /* access ARKodeARKStepMem structure */
    let retval = arkStep_AccessStepMem(ark_mem, "arkStep_SetOrder");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* set user-provided value, or default, depending on argument */
    {
        let mut step_mem = arkStep_mem_mut(ark_mem);
        if ord <= 0 {
            step_mem.q = Q_DEFAULT;
        } else {
            step_mem.q = ord;
        }

        /* clear Butcher tables, since user is requesting a change in method
        or a reset to defaults.  Tables will be set in ARKInitialSetup. */
        step_mem.stages = 0;
        step_mem.istage = 0;
        step_mem.p = 0;
    }

    let old_be = { arkStep_mem_mut(ark_mem).Be.take() };
    ARKodeButcherTable_Space(old_be.as_ref(), &mut Bliw, &mut Blrw);
    drop(old_be); /* ARKodeButcherTable_Free(step_mem->Be); step_mem->Be = NULL */
    {
        let mut m = ark_mem.borrow_mut();
        m.liw -= Bliw;
        m.lrw -= Blrw;
    }

    let old_bi = { arkStep_mem_mut(ark_mem).Bi.take() };
    ARKodeButcherTable_Space(old_bi.as_ref(), &mut Bliw, &mut Blrw);
    drop(old_bi); /* ARKodeButcherTable_Free(step_mem->Bi); step_mem->Bi = NULL */
    {
        let mut m = ark_mem.borrow_mut();
        m.liw -= Bliw;
        m.lrw -= Blrw;
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  arkStep_SetLinear:

  Specifies that the implicit portion of the problem is linear,
  and to tighten the linear solver tolerances while taking only
  one Newton iteration.  DO NOT USE IN COMBINATION WITH THE
  FIXED-POINT SOLVER.  Automatically tightens DeltaGammaMax
  to ensure that step size changes cause Jacobian recomputation.

  The argument should be 1 or 0, where 1 indicates that the
  Jacobian of fi with respect to y depends on time, and
  0 indicates that it is not time dependent.  Alternately, when
  using an iterative linear solver this flag denotes time
  dependence of the preconditioner.
  ---------------------------------------------------------------*/
pub fn arkStep_SetLinear(ark_mem: &ARKodeMem, timedepend: i32) -> i32 {
    /* access ARKodeARKStepMem structure */
    let retval = arkStep_AccessStepMem(ark_mem, "arkStep_SetLinear");
    if retval != ARK_SUCCESS {
        return retval;
    }

    let autonomous = { arkStep_mem_mut(ark_mem).autonomous };
    if (timedepend != 0) && autonomous {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "arkStep_SetLinear",
            file!(),
            "Incompatible settings, the problem is autonomous but the Jacobian is time dependent",
        );
        return ARK_ILL_INPUT;
    }

    /* set parameters */
    {
        let mut step_mem = arkStep_mem_mut(ark_mem);
        step_mem.linear = SUNTRUE;
        step_mem.linear_timedep = timedepend == 1;
        step_mem.dgmax = 100.0 * SUN_UNIT_ROUNDOFF;
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  arkStep_SetNonlinear:

  Specifies that the implicit portion of the problem is nonlinear.
  Used to undo a previous call to arkStep_SetLinear.  Automatically
  loosens DeltaGammaMax back to default value.
  ---------------------------------------------------------------*/
pub fn arkStep_SetNonlinear(ark_mem: &ARKodeMem) -> i32 {
    /* access ARKodeARKStepMem structure */
    let retval = arkStep_AccessStepMem(ark_mem, "arkStep_SetNonlinear");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* set parameters */
    {
        let mut step_mem = arkStep_mem_mut(ark_mem);
        step_mem.linear = SUNFALSE;
        step_mem.linear_timedep = SUNTRUE;
        step_mem.dgmax = DGMAX;
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  arkStep_SetAutonomous:

  Indicates if the problem is autonomous (True) or non-autonomous
  (False).
  ---------------------------------------------------------------*/
pub fn arkStep_SetAutonomous(ark_mem: &ARKodeMem, autonomous: sunbooleantype) -> i32 {
    /* access ARKodeARKStepMem structure */
    let mut retval = arkStep_AccessStepMem(ark_mem, "arkStep_SetAutonomous");
    if retval != ARK_SUCCESS {
        return retval;
    }

    {
        let mut step_mem = arkStep_mem_mut(ark_mem);
        step_mem.autonomous = autonomous;

        if autonomous && step_mem.linear {
            step_mem.linear_timedep = SUNFALSE;
        }
    }

    /* Reattach the nonlinear system function e.g., switching to/from an
       autonomous problem with the trivial predictor requires swapping the
       nonlinear system function provided to the nonlinear solver */
    retval = arkStep_SetNlsSysFn(ark_mem);
    if retval != ARK_SUCCESS {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "arkStep_SetAutonomous",
            file!(),
            "Setting nonlinear system function failed",
        );
        return ARK_ILL_INPUT;
    }

    /* This will be better handled when the temp vector stack is added */
    if autonomous {
        /* Allocate tempv5 if needed */
        let yn = { ark_mem.borrow().yn.clone() }.expect("yn");
        let mut tempv5 = ark_mem.borrow_mut().tempv5.take();
        let ok = arkAllocVec(ark_mem, &yn, &mut tempv5);
        ark_mem.borrow_mut().tempv5 = tempv5;
        if !ok {
            arkProcessError(
                Some(ark_mem),
                ARK_MEM_FAIL,
                line!() as i32,
                "arkStep_SetAutonomous",
                file!(),
                MSG_ARK_MEM_FAIL,
            );
            return ARK_MEM_FAIL;
        }
    } else {
        /* Free tempv5 if necessary */
        let mut tempv5 = ark_mem.borrow_mut().tempv5.take();
        arkFreeVec(ark_mem, &mut tempv5);
        ark_mem.borrow_mut().tempv5 = tempv5;
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  arkStep_SetNonlinCRDown:

  Specifies the user-provided nonlinear convergence constant
  crdown.  Legal values are strictly positive; illegal values
  imply a reset to the default.
  ---------------------------------------------------------------*/
pub fn arkStep_SetNonlinCRDown(ark_mem: &ARKodeMem, crdown: sunrealtype) -> i32 {
    /* access ARKodeARKStepMem structure */
    let retval = arkStep_AccessStepMem(ark_mem, "arkStep_SetNonlinCRDown");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* if argument legal set it, otherwise set default */
    {
        let mut step_mem = arkStep_mem_mut(ark_mem);
        if crdown <= ZERO {
            step_mem.crdown = CRDOWN;
        } else {
            step_mem.crdown = crdown;
        }
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  arkStep_SetNonlinRDiv:

  Specifies the user-provided nonlinear convergence constant
  rdiv.  Legal values are strictly positive; illegal values
  imply a reset to the default.
  ---------------------------------------------------------------*/
pub fn arkStep_SetNonlinRDiv(ark_mem: &ARKodeMem, rdiv: sunrealtype) -> i32 {
    /* access ARKodeARKStepMem structure */
    let retval = arkStep_AccessStepMem(ark_mem, "arkStep_SetNonlinRDiv");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* if argument legal set it, otherwise set default */
    {
        let mut step_mem = arkStep_mem_mut(ark_mem);
        if rdiv <= ZERO {
            step_mem.rdiv = RDIV;
        } else {
            step_mem.rdiv = rdiv;
        }
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  arkStep_SetDeltaGammaMax:

  Specifies the user-provided linear setup decision constant
  dgmax.  Legal values are strictly positive; illegal values imply
  a reset to the default.
  ---------------------------------------------------------------*/
pub fn arkStep_SetDeltaGammaMax(ark_mem: &ARKodeMem, dgmax: sunrealtype) -> i32 {
    /* access ARKodeARKStepMem structure */
    let retval = arkStep_AccessStepMem(ark_mem, "arkStep_SetDeltaGammaMax");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* if argument legal set it, otherwise set default */
    {
        let mut step_mem = arkStep_mem_mut(ark_mem);
        if dgmax <= ZERO {
            step_mem.dgmax = DGMAX;
        } else {
            step_mem.dgmax = dgmax;
        }
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  arkStep_SetLSetupFrequency:

  Specifies the user-provided linear setup decision constant
  msbp.  Positive values give the frequency for calling lsetup;
  negative values imply recomputation of lsetup at each nonlinear
  solve; a zero value implies a reset to the default.
  ---------------------------------------------------------------*/
pub fn arkStep_SetLSetupFrequency(ark_mem: &ARKodeMem, msbp: i32) -> i32 {
    /* access ARKodeARKStepMem structure */
    let retval = arkStep_AccessStepMem(ark_mem, "arkStep_SetLSetupFrequency");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* if argument legal set it, otherwise set default */
    {
        let mut step_mem = arkStep_mem_mut(ark_mem);
        if msbp == 0 {
            step_mem.msbp = MSBP;
        } else {
            step_mem.msbp = msbp;
        }
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  arkStep_SetPredictorMethod:

  Specifies the method to use for predicting implicit solutions.
  Non-default choices are {1,2,3,4}, all others will use default
  (trivial) predictor.
  ---------------------------------------------------------------*/
pub fn arkStep_SetPredictorMethod(ark_mem: &ARKodeMem, pred_method: i32) -> i32 {
    /* access ARKodeARKStepMem structure */
    let mut retval = arkStep_AccessStepMem(ark_mem, "arkStep_SetPredictorMethod");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* set parameter */
    arkStep_mem_mut(ark_mem).predictor = pred_method;

    /* Reattach the nonlinear system function e.g., switching to/from the trivial
       predictor with an autonomous problem requires swapping the nonlinear system
       function provided to the nonlinear solver */
    retval = arkStep_SetNlsSysFn(ark_mem);
    if retval != ARK_SUCCESS {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "arkStep_SetPredictorMethod",
            file!(),
            "Setting nonlinear system function failed",
        );
        return ARK_ILL_INPUT;
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  arkStep_SetMaxNonlinIters:

  Specifies the maximum number of nonlinear iterations during
  one solve.  A non-positive input implies a reset to the
  default value.
  ---------------------------------------------------------------*/
pub fn arkStep_SetMaxNonlinIters(ark_mem: &ARKodeMem, maxcor: i32) -> i32 {
    /* access ARKodeARKStepMem structure */
    let mut retval = arkStep_AccessStepMem(ark_mem, "arkStep_SetMaxNonlinIters");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* SUNFunctionBegin(ark_mem->sunctx): error checks are OFF in this build */

    /* Return error message if no NLS module is present */
    let NLS = { arkStep_mem_mut(ark_mem).NLS.clone() };
    if NLS.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_NLS_OP_ERR,
            line!() as i32,
            "arkStep_SetMaxNonlinIters",
            file!(),
            "No SUNNonlinearSolver object is present",
        );
        return ARK_ILL_INPUT;
    }
    let NLS = NLS.expect("NLS");

    /* argument <= 0 sets default, otherwise set input */
    let new_maxcor = {
        let mut step_mem = arkStep_mem_mut(ark_mem);
        if maxcor <= 0 {
            step_mem.maxcor = MAXCOR;
        } else {
            step_mem.maxcor = maxcor;
        }
        step_mem.maxcor
    };

    /* send argument to NLS structure */
    retval = SUNNonlinSolSetMaxIters(&NLS, new_maxcor);
    if retval != SUN_SUCCESS {
        arkProcessError(
            Some(ark_mem),
            ARK_NLS_OP_ERR,
            line!() as i32,
            "arkStep_SetMaxNonlinIters",
            file!(),
            "Error setting maxcor in SUNNonlinearSolver object",
        );
        return ARK_NLS_OP_ERR;
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  arkStep_SetNonlinConvCoef:

  Specifies the coefficient in the nonlinear solver convergence
  test.  A non-positive input implies a reset to the default value.
  ---------------------------------------------------------------*/
pub fn arkStep_SetNonlinConvCoef(ark_mem: &ARKodeMem, nlscoef: sunrealtype) -> i32 {
    /* access ARKodeARKStepMem structure */
    let retval = arkStep_AccessStepMem(ark_mem, "arkStep_SetNonlinConvCoef");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* argument <= 0 sets default, otherwise set input */
    {
        let mut step_mem = arkStep_mem_mut(ark_mem);
        if nlscoef <= ZERO {
            step_mem.nlscoef = NLSCOEF;
        } else {
            step_mem.nlscoef = nlscoef;
        }
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  arkStep_SetStagePredictFn:  Specifies a user-provided step
  predictor function having type ARKStagePredictFn.  A
  NULL input function disables calls to this routine.
  ---------------------------------------------------------------*/
pub fn arkStep_SetStagePredictFn(
    ark_mem: &ARKodeMem,
    PredictStage: Option<ARKStagePredictFn>,
) -> i32 {
    /* access ARKodeARKStepMem structure and set function pointer */
    let retval = arkStep_AccessStepMem(ark_mem, "arkStep_SetStagePredictFn");
    if retval != ARK_SUCCESS {
        return retval;
    }

    arkStep_mem_mut(ark_mem).stage_predict = PredictStage;
    ARK_SUCCESS
}

/*---------------------------------------------------------------
  arkStep_SetDeduceImplicitRhs:

  Specifies if an optimization is used to avoid an evaluation of
  fi after a nonlinear solve for an implicit stage.  If stage
  postprocessecing in enabled, this option is ignored, and fi is
  never deduced.

  An argument of SUNTRUE indicates that fi is deduced to compute
  fi(z_i), and SUNFALSE indicates that fi(z_i) is computed with
  an additional evaluation of fi.
  ---------------------------------------------------------------*/
pub fn arkStep_SetDeduceImplicitRhs(ark_mem: &ARKodeMem, deduce: sunbooleantype) -> i32 {
    /* access ARKodeARKStepMem structure and set function pointer */
    let retval = arkStep_AccessStepMem(ark_mem, "arkStep_SetDeduceImplicitRhs");
    if retval != ARK_SUCCESS {
        return retval;
    }

    arkStep_mem_mut(ark_mem).deduce_rhs = deduce;
    ARK_SUCCESS
}

/*---------------------------------------------------------------
  arkStep_GetCurrentGamma: Returns the current value of gamma
  ---------------------------------------------------------------*/
pub fn arkStep_GetCurrentGamma(ark_mem: &ARKodeMem, gamma: &mut sunrealtype) -> i32 {
    let retval = arkStep_AccessStepMem(ark_mem, "arkStep_GetCurrentGamma");
    if retval != ARK_SUCCESS {
        return retval;
    }
    *gamma = arkStep_mem_mut(ark_mem).gamma;
    retval
}

/*---------------------------------------------------------------
  arkStep_GetEstLocalErrors: Returns the current local truncation
  error estimate vector
  ---------------------------------------------------------------*/
pub fn arkStep_GetEstLocalErrors(ark_mem: &ARKodeMem, ele: &N_Vector) -> i32 {
    let retval = arkStep_AccessStepMem(ark_mem, "arkStep_GetEstLocalErrors");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* return an error if local truncation error is not computed */
    let (fixedstep, accum_error_type, tempv1) = {
        let m = ark_mem.borrow();
        (
            m.fixedstep,
            m.AccumErrorType,
            m.tempv1.clone().expect("tempv1"),
        )
    };
    let p = { arkStep_mem_mut(ark_mem).p };
    if (fixedstep && (accum_error_type == ARK_ACCUMERROR_NONE)) || (p <= 0) {
        return ARK_STEPPER_UNSUPPORTED;
    }

    /* otherwise, copy local truncation error vector to output */
    N_VScale(ONE, &tempv1, ele);
    ARK_SUCCESS
}

/*---------------------------------------------------------------
  arkStep_GetNumLinSolvSetups:

  Returns the current number of calls to the lsetup routine
  ---------------------------------------------------------------*/
pub fn arkStep_GetNumLinSolvSetups(ark_mem: &ARKodeMem, nlinsetups: &mut i64) -> i32 {
    /* access ARKodeARKStepMem structure */
    let retval = arkStep_AccessStepMem(ark_mem, "arkStep_GetNumLinSolvSetups");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* get value from step_mem */
    *nlinsetups = arkStep_mem_mut(ark_mem).nsetups;

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  arkStep_GetNumNonlinSolvIters:

  Returns the current number of nonlinear solver iterations
  ---------------------------------------------------------------*/
pub fn arkStep_GetNumNonlinSolvIters(ark_mem: &ARKodeMem, nniters: &mut i64) -> i32 {
    /* access ARKodeARKStepMem structure */
    let retval = arkStep_AccessStepMem(ark_mem, "arkStep_GetNumNonlinSolvIters");
    if retval != ARK_SUCCESS {
        return retval;
    }

    *nniters = arkStep_mem_mut(ark_mem).nls_iters;

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  arkStep_GetNumNonlinSolvConvFails:

  Returns the current number of nonlinear solver convergence fails
  ---------------------------------------------------------------*/
pub fn arkStep_GetNumNonlinSolvConvFails(ark_mem: &ARKodeMem, nnfails: &mut i64) -> i32 {
    /* access ARKodeARKStepMem structure */
    let retval = arkStep_AccessStepMem(ark_mem, "arkStep_GetNumNonlinSolvConvFails");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* set output from step_mem */
    *nnfails = arkStep_mem_mut(ark_mem).nls_fails;

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  arkStep_GetNonlinSolvStats:

  Returns nonlinear solver statistics
  ---------------------------------------------------------------*/
pub fn arkStep_GetNonlinSolvStats(
    ark_mem: &ARKodeMem,
    nniters: &mut i64,
    nnfails: &mut i64,
) -> i32 {
    /* access ARKodeARKStepMem structure */
    let retval = arkStep_AccessStepMem(ark_mem, "arkStep_GetNonlinSolvStats");
    if retval != ARK_SUCCESS {
        return retval;
    }

    {
        let step_mem = arkStep_mem_mut(ark_mem);
        *nniters = step_mem.nls_iters;
        *nnfails = step_mem.nls_fails;
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  arkStep_GetStageIndex:

  Returns the current stage index and number of stages
  ---------------------------------------------------------------*/
pub fn arkStep_GetStageIndex(ark_mem: &ARKodeMem, stage: &mut i32, max_stages: &mut i32) -> i32 {
    /* access ARKodeARKStepMem structure */
    let retval = arkStep_AccessStepMem(ark_mem, "arkStep_GetStageIndex");
    if retval != ARK_SUCCESS {
        return retval;
    }

    {
        let step_mem = arkStep_mem_mut(ark_mem);
        *stage = step_mem.istage;
        *max_stages = step_mem.stages;
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  arkStep_PrintAllStats:

  Prints integrator statistics
  ---------------------------------------------------------------*/
pub fn arkStep_PrintAllStats(
    ark_mem: &ARKodeMem,
    outfile: &SUNFile,
    fmt: SUNOutputFormat,
) -> i32 {
    /* access ARKodeARKStepMem structure */
    let retval = arkStep_AccessStepMem(ark_mem, "arkStep_PrintAllStats");
    if retval != ARK_SUCCESS {
        return retval;
    }

    let (nfe, nfi, nls_iters, nls_fails, nsetups) = {
        let step_mem = arkStep_mem_mut(ark_mem);
        (
            step_mem.nfe,
            step_mem.nfi,
            step_mem.nls_iters,
            step_mem.nls_fails,
            step_mem.nsetups,
        )
    };
    let nst = ark_mem.borrow().nst;

    /* function evaluations */
    sunfprintf_long(outfile, fmt, SUNFALSE, "Explicit RHS fn evals", nfe);
    sunfprintf_long(outfile, fmt, SUNFALSE, "Implicit RHS fn evals", nfi);

    /* nonlinear solver stats */
    sunfprintf_long(outfile, fmt, SUNFALSE, "NLS iters", nls_iters);
    sunfprintf_long(outfile, fmt, SUNFALSE, "NLS fails", nls_fails);
    if nst > 0 {
        sunfprintf_real(
            outfile,
            fmt,
            SUNFALSE,
            "NLS iters per step",
            nls_iters as sunrealtype / nst as sunrealtype,
        );
    }

    /* linear solver stats */
    sunfprintf_long(outfile, fmt, SUNFALSE, "LS setups", nsetups);
    let step_getlinmem = { ark_mem.borrow().step_getlinmem }.expect("step_getlinmem");
    if step_getlinmem(ark_mem) {
        let (nje, nfeDQ, npe, nps, nli, ncfl, njtsetup, njtimes) = {
            let arkls_mem = arkls_mem_mut(ark_mem);
            (
                arkls_mem.nje,
                arkls_mem.nfeDQ,
                arkls_mem.npe,
                arkls_mem.nps,
                arkls_mem.nli,
                arkls_mem.ncfl,
                arkls_mem.njtsetup,
                arkls_mem.njtimes,
            )
        };
        sunfprintf_long(outfile, fmt, SUNFALSE, "Jac fn evals", nje);
        sunfprintf_long(outfile, fmt, SUNFALSE, "LS RHS fn evals", nfeDQ);
        sunfprintf_long(outfile, fmt, SUNFALSE, "Prec setup evals", npe);
        sunfprintf_long(outfile, fmt, SUNFALSE, "Prec solves", nps);
        sunfprintf_long(outfile, fmt, SUNFALSE, "LS iters", nli);
        sunfprintf_long(outfile, fmt, SUNFALSE, "LS fails", ncfl);
        sunfprintf_long(outfile, fmt, SUNFALSE, "Jac-times setups", njtsetup);
        sunfprintf_long(outfile, fmt, SUNFALSE, "Jac-times evals", njtimes);
        if nls_iters > 0 {
            sunfprintf_real(
                outfile,
                fmt,
                SUNFALSE,
                "LS iters per NLS iter",
                nli as sunrealtype / nls_iters as sunrealtype,
            );
            sunfprintf_real(
                outfile,
                fmt,
                SUNFALSE,
                "Jac evals per NLS iter",
                nje as sunrealtype / nls_iters as sunrealtype,
            );
            sunfprintf_real(
                outfile,
                fmt,
                SUNFALSE,
                "Prec evals per NLS iter",
                npe as sunrealtype / nls_iters as sunrealtype,
            );
        }
    }

    /* mass solve stats */
    let step_getmassmem = { ark_mem.borrow().step_getmassmem }.expect("step_getmassmem");
    if step_getmassmem(ark_mem) {
        let (nmsetups, nmsolves, npe, nps, nli, ncfl, nmtsetup, nmtimes) = {
            let arklsm_mem = arkls_mass_mem_mut(ark_mem);
            (
                arklsm_mem.nmsetups,
                arklsm_mem.nmsolves,
                arklsm_mem.npe,
                arklsm_mem.nps,
                arklsm_mem.nli,
                arklsm_mem.ncfl,
                arklsm_mem.nmtsetup,
                arklsm_mem.nmtimes,
            )
        };
        sunfprintf_long(outfile, fmt, SUNFALSE, "Mass setups", nmsetups);
        sunfprintf_long(outfile, fmt, SUNFALSE, "Mass solves", nmsolves);
        sunfprintf_long(outfile, fmt, SUNFALSE, "Mass Prec setup evals", npe);
        sunfprintf_long(outfile, fmt, SUNFALSE, "Mass Prec solves", nps);
        sunfprintf_long(outfile, fmt, SUNFALSE, "Mass LS iters", nli);
        sunfprintf_long(outfile, fmt, SUNFALSE, "Mass LS fails", ncfl);
        sunfprintf_long(outfile, fmt, SUNFALSE, "Mass-times setups", nmtsetup);
        sunfprintf_long(outfile, fmt, SUNFALSE, "Mass-times evals", nmtimes);
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  arkStep_WriteParameters:

  Outputs all solver parameters to the provided file pointer.
  ---------------------------------------------------------------*/
pub fn arkStep_WriteParameters(ark_mem: &ARKodeMem, fp: &SUNFile) -> i32 {
    /* access ARKodeARKStepMem structure */
    let retval = arkStep_AccessStepMem(ark_mem, "arkStep_WriteParameters");
    if retval != ARK_SUCCESS {
        return retval;
    }

    let (q, linear, linear_timedep, explicit, implicit, predictor, nlscoef, maxcor, crdown, rdiv, dgmax, msbp) = {
        let step_mem = arkStep_mem_mut(ark_mem);
        (
            step_mem.q,
            step_mem.linear,
            step_mem.linear_timedep,
            step_mem.explicit,
            step_mem.implicit,
            step_mem.predictor,
            step_mem.nlscoef,
            step_mem.maxcor,
            step_mem.crdown,
            step_mem.rdiv,
            step_mem.dgmax,
            step_mem.msbp,
        )
    };

    /* print integrator parameters to file */
    fp.write_str("ARKStep time step module parameters:\n");
    fp.write_str(&format!("  Method order {}\n", q));
    if linear {
        fp.write_str("  Linear implicit problem");
        if linear_timedep {
            fp.write_str(" (time-dependent Jacobian)\n");
        } else {
            fp.write_str(" (time-independent Jacobian)\n");
        }
    }
    if explicit && implicit {
        fp.write_str("  ImEx integrator\n");
    } else if implicit {
        fp.write_str("  Implicit integrator\n");
    } else {
        fp.write_str("  Explicit integrator\n");
    }

    if implicit {
        fp.write_str(&format!("  Implicit predictor method = {}\n", predictor));
        fp.write_str(&format!(
            "  Implicit solver tolerance coefficient = {}\n",
            sun_format_g(nlscoef)
        ));
        fp.write_str(&format!(
            "  Maximum number of nonlinear corrections = {}\n",
            maxcor
        ));
        fp.write_str(&format!(
            "  Nonlinear convergence rate constant = {}\n",
            sun_format_g(crdown)
        ));
        fp.write_str(&format!(
            "  Nonlinear divergence tolerance = {}\n",
            sun_format_g(rdiv)
        ));
        fp.write_str(&format!(
            "  Gamma factor LSetup tolerance = {}\n",
            sun_format_g(dgmax)
        ));
        fp.write_str(&format!(
            "  Number of steps between LSetup calls = {}\n",
            msbp
        ));
    }
    fp.write_str("\n");

    ARK_SUCCESS
}

/*===============================================================
  Exported-but-deprecated user-callable functions.
  ===============================================================*/

pub fn ARKStepCreateMRIStepInnerStepper(
    inner_arkode_mem: &ARKodeMem,
    stepper: &mut Option<MRIStepInnerStepper>,
) -> i32 {
    ARKodeCreateMRIStepInnerStepper(inner_arkode_mem, stepper)
}

pub fn ARKStepResize(
    arkode_mem: &ARKodeMem,
    y0: &N_Vector,
    hscale: sunrealtype,
    t0: sunrealtype,
    resize: Option<ARKVecResizeFn>,
    resize_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    ARKodeResize(arkode_mem, y0, hscale, t0, resize, resize_data)
}

pub fn ARKStepReset(arkode_mem: &ARKodeMem, tR: sunrealtype, yR: &N_Vector) -> i32 {
    ARKodeReset(arkode_mem, tR, yR)
}

pub fn ARKStepSStolerances(
    arkode_mem: &ARKodeMem,
    reltol: sunrealtype,
    abstol: sunrealtype,
) -> i32 {
    ARKodeSStolerances(arkode_mem, reltol, abstol)
}

pub fn ARKStepSVtolerances(
    arkode_mem: &ARKodeMem,
    reltol: sunrealtype,
    abstol: &N_Vector,
) -> i32 {
    ARKodeSVtolerances(arkode_mem, reltol, abstol)
}

pub fn ARKStepWFtolerances(arkode_mem: &ARKodeMem, efun: ARKEwtFn) -> i32 {
    ARKodeWFtolerances(arkode_mem, efun)
}

pub fn ARKStepResStolerance(arkode_mem: &ARKodeMem, rabstol: sunrealtype) -> i32 {
    ARKodeResStolerance(arkode_mem, rabstol)
}

pub fn ARKStepResVtolerance(arkode_mem: &ARKodeMem, rabstol: &N_Vector) -> i32 {
    ARKodeResVtolerance(arkode_mem, rabstol)
}

pub fn ARKStepResFtolerance(arkode_mem: &ARKodeMem, rfun: ARKRwtFn) -> i32 {
    ARKodeResFtolerance(arkode_mem, rfun)
}

pub fn ARKStepSetLinearSolver(
    arkode_mem: &ARKodeMem,
    LS: &SUNLinearSolver,
    A: Option<&SUNMatrix>,
) -> i32 {
    ARKodeSetLinearSolver(arkode_mem, LS, A)
}

pub fn ARKStepSetMassLinearSolver(
    arkode_mem: &ARKodeMem,
    LS: &SUNLinearSolver,
    M: Option<&SUNMatrix>,
    time_dep: sunbooleantype,
) -> i32 {
    ARKodeSetMassLinearSolver(arkode_mem, LS, M, time_dep)
}

pub fn ARKStepRootInit(arkode_mem: &ARKodeMem, nrtfn: i32, g: Option<ARKRootFn>) -> i32 {
    ARKodeRootInit(arkode_mem, nrtfn, g)
}

pub fn ARKStepSetDefaults(arkode_mem: &ARKodeMem) -> i32 {
    ARKodeSetDefaults(arkode_mem)
}

pub fn ARKStepSetOptimalParams(arkode_mem: &ARKodeMem) -> i32 {
    /* TODO: do we need to do something here? This is deprecated with no
     * ARKodeSetOptimalParams to replace it */
    let ark_mem = arkode_mem;
    let mut retval;
    let mut lenrw: i64 = 0;
    let mut leniw: i64 = 0;

    /* access ARKodeMem and ARKodeARKStepMem structures */
    retval = arkStep_AccessARKODEStepMem(arkode_mem, "ARKStepSetOptimalParams");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* access ARKodeHAdaptMem structure */
    if ark_mem.borrow().hadapt_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "ARKStepSetOptimalParams",
            file!(),
            MSG_ARKADAPT_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    /* Remove current SUNAdaptController object */
    let hcontroller = {
        ark_mem
            .borrow()
            .hadapt_mem
            .as_ref()
            .expect("hadapt_mem")
            .hcontroller
            .clone()
    }
    .expect("hcontroller");
    retval = SUNAdaptController_Space(&hcontroller, &mut lenrw, &mut leniw);
    if retval == SUN_SUCCESS {
        let mut m = ark_mem.borrow_mut();
        m.liw -= leniw;
        m.lrw -= lenrw;
    }
    let owncontroller = {
        ark_mem
            .borrow()
            .hadapt_mem
            .as_ref()
            .expect("hadapt_mem")
            .owncontroller
    };
    if owncontroller {
        retval = SUNAdaptController_Destroy(Some(hcontroller));
        ark_mem
            .borrow_mut()
            .hadapt_mem
            .as_mut()
            .expect("hadapt_mem")
            .owncontroller = SUNFALSE;
        if retval != SUN_SUCCESS {
            arkProcessError(
                Some(ark_mem),
                ARK_MEM_FAIL,
                line!() as i32,
                "ARKStepSetOptimalParams",
                file!(),
                "SUNAdaptController_Destroy failure",
            );
            return ARK_MEM_FAIL;
        }
    }
    ark_mem
        .borrow_mut()
        .hadapt_mem
        .as_mut()
        .expect("hadapt_mem")
        .hcontroller = None;

    /* Choose values based on method, order */

    let (explicit, implicit, q) = {
        let step_mem = arkStep_mem_mut(ark_mem);
        (step_mem.explicit, step_mem.implicit, step_mem.q)
    };
    let sunctx = ark_mem.borrow().sunctx.clone();

    /*    explicit */
    if explicit && !implicit {
        let C = SUNAdaptController_PI(&sunctx);
        ark_mem
            .borrow_mut()
            .hadapt_mem
            .as_mut()
            .expect("hadapt_mem")
            .hcontroller = C.clone();
        if C.is_none() {
            arkProcessError(
                Some(ark_mem),
                ARK_MEM_FAIL,
                line!() as i32,
                "ARKStepSetOptimalParams",
                file!(),
                "SUNAdaptController_PI allocation failure",
            );
            return ARK_MEM_FAIL;
        }
        let C = C.expect("hcontroller");
        let _ = SUNAdaptController_SetErrorBias(&C, 1.2);
        let _ = SUNAdaptController_SetParams_PI(&C, 0.8, -0.31);
        {
            let mut m = ark_mem.borrow_mut();
            let hadapt_mem = m.hadapt_mem.as_mut().expect("hadapt_mem");
            hadapt_mem.safety = 0.99;
            hadapt_mem.growth = 25.0;
            hadapt_mem.etamxf = 0.3;
            hadapt_mem.pq = PQ;
        }

    /*    implicit */
    } else if implicit && !explicit {
        match q {
            2 => {
                /* just use standard defaults since better ones unknown */
                let C = SUNAdaptController_PID(&sunctx);
                ark_mem
                    .borrow_mut()
                    .hadapt_mem
                    .as_mut()
                    .expect("hadapt_mem")
                    .hcontroller = C.clone();
                if C.is_none() {
                    arkProcessError(
                        Some(ark_mem),
                        ARK_MEM_FAIL,
                        line!() as i32,
                        "ARKStepSetOptimalParams",
                        file!(),
                        "SUNAdaptController_PID allocation failure",
                    );
                    return ARK_MEM_FAIL;
                }
                {
                    let mut m = ark_mem.borrow_mut();
                    let hadapt_mem = m.hadapt_mem.as_mut().expect("hadapt_mem");
                    hadapt_mem.safety = SAFETY;
                    hadapt_mem.growth = GROWTH;
                    hadapt_mem.etamxf = ETAMXF;
                    hadapt_mem.small_nef = SMALL_NEF;
                    hadapt_mem.etacf = ETACF;
                    hadapt_mem.pq = PQ;
                }
                {
                    let mut step_mem = arkStep_mem_mut(ark_mem);
                    step_mem.nlscoef = 0.001;
                    step_mem.maxcor = 5;
                    step_mem.crdown = CRDOWN;
                    step_mem.rdiv = RDIV;
                    step_mem.dgmax = DGMAX;
                    step_mem.msbp = MSBP;
                }
            }
            3 => {
                let C = SUNAdaptController_I(&sunctx);
                ark_mem
                    .borrow_mut()
                    .hadapt_mem
                    .as_mut()
                    .expect("hadapt_mem")
                    .hcontroller = C.clone();
                if C.is_none() {
                    arkProcessError(
                        Some(ark_mem),
                        ARK_MEM_FAIL,
                        line!() as i32,
                        "ARKStepSetOptimalParams",
                        file!(),
                        "SUNAdaptController_I allocation failure",
                    );
                    return ARK_MEM_FAIL;
                }
                let C = C.expect("hcontroller");
                let _ = SUNAdaptController_SetErrorBias(&C, 1.9);
                {
                    let mut m = ark_mem.borrow_mut();
                    let hadapt_mem = m.hadapt_mem.as_mut().expect("hadapt_mem");
                    hadapt_mem.safety = 0.957;
                    hadapt_mem.growth = 17.6;
                    hadapt_mem.etamxf = 0.45;
                    hadapt_mem.small_nef = SMALL_NEF;
                    hadapt_mem.etacf = ETACF;
                    hadapt_mem.pq = PQ;
                }
                {
                    let mut step_mem = arkStep_mem_mut(ark_mem);
                    step_mem.nlscoef = 0.22;
                    step_mem.crdown = 0.17;
                    step_mem.rdiv = 2.3;
                    step_mem.dgmax = 0.19;
                    step_mem.msbp = 60;
                }
            }
            4 => {
                let C = SUNAdaptController_PID(&sunctx);
                ark_mem
                    .borrow_mut()
                    .hadapt_mem
                    .as_mut()
                    .expect("hadapt_mem")
                    .hcontroller = C.clone();
                if C.is_none() {
                    arkProcessError(
                        Some(ark_mem),
                        ARK_MEM_FAIL,
                        line!() as i32,
                        "ARKStepSetOptimalParams",
                        file!(),
                        "SUNAdaptController_PID allocation failure",
                    );
                    return ARK_MEM_FAIL;
                }
                let C = C.expect("hcontroller");
                let _ = SUNAdaptController_SetErrorBias(&C, 1.2);
                let _ = SUNAdaptController_SetParams_PID(&C, 0.535, -0.209, 0.148);
                {
                    let mut m = ark_mem.borrow_mut();
                    let hadapt_mem = m.hadapt_mem.as_mut().expect("hadapt_mem");
                    hadapt_mem.safety = 0.988;
                    hadapt_mem.growth = 31.5;
                    hadapt_mem.etamxf = 0.33;
                    hadapt_mem.small_nef = SMALL_NEF;
                    hadapt_mem.etacf = ETACF;
                    hadapt_mem.pq = PQ;
                }
                {
                    let mut step_mem = arkStep_mem_mut(ark_mem);
                    step_mem.nlscoef = 0.24;
                    step_mem.crdown = 0.26;
                    step_mem.rdiv = 2.3;
                    step_mem.dgmax = 0.16;
                    step_mem.msbp = 31;
                }
            }
            5 => {
                let C = SUNAdaptController_PID(&sunctx);
                ark_mem
                    .borrow_mut()
                    .hadapt_mem
                    .as_mut()
                    .expect("hadapt_mem")
                    .hcontroller = C.clone();
                if C.is_none() {
                    arkProcessError(
                        Some(ark_mem),
                        ARK_MEM_FAIL,
                        line!() as i32,
                        "ARKStepSetOptimalParams",
                        file!(),
                        "SUNAdaptController_PID allocation failure",
                    );
                    return ARK_MEM_FAIL;
                }
                let C = C.expect("hcontroller");
                let _ = SUNAdaptController_SetErrorBias(&C, 3.3);
                let _ = SUNAdaptController_SetParams_PID(&C, 0.56, -0.338, 0.14);
                {
                    let mut m = ark_mem.borrow_mut();
                    let hadapt_mem = m.hadapt_mem.as_mut().expect("hadapt_mem");
                    hadapt_mem.safety = 0.937;
                    hadapt_mem.growth = 22.0;
                    hadapt_mem.etamxf = 0.44;
                    hadapt_mem.small_nef = SMALL_NEF;
                    hadapt_mem.etacf = ETACF;
                    hadapt_mem.pq = PQ;
                }
                {
                    let mut step_mem = arkStep_mem_mut(ark_mem);
                    step_mem.nlscoef = 0.25;
                    step_mem.crdown = 0.4;
                    step_mem.rdiv = 2.3;
                    step_mem.dgmax = 0.32;
                    step_mem.msbp = 31;
                }
            }
            _ => {}
        }

    /*    imex */
    } else {
        match q {
            2 => {
                /* just use standard defaults since better ones unknown */
                let C = SUNAdaptController_PID(&sunctx);
                ark_mem
                    .borrow_mut()
                    .hadapt_mem
                    .as_mut()
                    .expect("hadapt_mem")
                    .hcontroller = C.clone();
                if C.is_none() {
                    arkProcessError(
                        Some(ark_mem),
                        ARK_MEM_FAIL,
                        line!() as i32,
                        "ARKStepSetOptimalParams",
                        file!(),
                        "SUNAdaptController_PID allocation failure",
                    );
                    return ARK_MEM_FAIL;
                }
                {
                    let mut m = ark_mem.borrow_mut();
                    let hadapt_mem = m.hadapt_mem.as_mut().expect("hadapt_mem");
                    hadapt_mem.safety = SAFETY;
                    hadapt_mem.growth = GROWTH;
                    hadapt_mem.etamxf = ETAMXF;
                    hadapt_mem.small_nef = SMALL_NEF;
                    hadapt_mem.etacf = ETACF;
                    hadapt_mem.pq = PQ;
                }
                {
                    let mut step_mem = arkStep_mem_mut(ark_mem);
                    step_mem.nlscoef = 0.001;
                    step_mem.maxcor = 5;
                    step_mem.crdown = CRDOWN;
                    step_mem.rdiv = RDIV;
                    step_mem.dgmax = DGMAX;
                    step_mem.msbp = MSBP;
                }
            }
            3 => {
                let C = SUNAdaptController_PID(&sunctx);
                ark_mem
                    .borrow_mut()
                    .hadapt_mem
                    .as_mut()
                    .expect("hadapt_mem")
                    .hcontroller = C.clone();
                if C.is_none() {
                    arkProcessError(
                        Some(ark_mem),
                        ARK_MEM_FAIL,
                        line!() as i32,
                        "ARKStepSetOptimalParams",
                        file!(),
                        "SUNAdaptController_PID allocation failure",
                    );
                    return ARK_MEM_FAIL;
                }
                let C = C.expect("hcontroller");
                let _ = SUNAdaptController_SetErrorBias(&C, 1.42);
                let _ = SUNAdaptController_SetParams_PID(&C, 0.54, -0.36, 0.14);
                {
                    let mut m = ark_mem.borrow_mut();
                    let hadapt_mem = m.hadapt_mem.as_mut().expect("hadapt_mem");
                    hadapt_mem.safety = 0.965;
                    hadapt_mem.growth = 28.7;
                    hadapt_mem.etamxf = 0.46;
                    hadapt_mem.small_nef = SMALL_NEF;
                    hadapt_mem.etacf = ETACF;
                    hadapt_mem.pq = PQ;
                }
                {
                    let mut step_mem = arkStep_mem_mut(ark_mem);
                    step_mem.nlscoef = 0.22;
                    step_mem.crdown = 0.17;
                    step_mem.rdiv = 2.3;
                    step_mem.dgmax = 0.19;
                    step_mem.msbp = 60;
                }
            }
            4 => {
                let C = SUNAdaptController_PID(&sunctx);
                ark_mem
                    .borrow_mut()
                    .hadapt_mem
                    .as_mut()
                    .expect("hadapt_mem")
                    .hcontroller = C.clone();
                if C.is_none() {
                    arkProcessError(
                        Some(ark_mem),
                        ARK_MEM_FAIL,
                        line!() as i32,
                        "ARKStepSetOptimalParams",
                        file!(),
                        "SUNAdaptController_PID allocation failure",
                    );
                    return ARK_MEM_FAIL;
                }
                let C = C.expect("hcontroller");
                let _ = SUNAdaptController_SetErrorBias(&C, 1.35);
                let _ = SUNAdaptController_SetParams_PID(&C, 0.543, -0.297, 0.14);
                {
                    let mut m = ark_mem.borrow_mut();
                    let hadapt_mem = m.hadapt_mem.as_mut().expect("hadapt_mem");
                    hadapt_mem.safety = 0.97;
                    hadapt_mem.growth = 25.0;
                    hadapt_mem.etamxf = 0.47;
                    hadapt_mem.small_nef = SMALL_NEF;
                    hadapt_mem.etacf = ETACF;
                    hadapt_mem.pq = PQ;
                }
                {
                    let mut step_mem = arkStep_mem_mut(ark_mem);
                    step_mem.nlscoef = 0.24;
                    step_mem.crdown = 0.26;
                    step_mem.rdiv = 2.3;
                    step_mem.dgmax = 0.16;
                    step_mem.msbp = 31;
                }
            }
            5 => {
                let C = SUNAdaptController_PI(&sunctx);
                ark_mem
                    .borrow_mut()
                    .hadapt_mem
                    .as_mut()
                    .expect("hadapt_mem")
                    .hcontroller = C.clone();
                if C.is_none() {
                    arkProcessError(
                        Some(ark_mem),
                        ARK_MEM_FAIL,
                        line!() as i32,
                        "ARKStepSetOptimalParams",
                        file!(),
                        "SUNAdaptController_PI allocation failure",
                    );
                    return ARK_MEM_FAIL;
                }
                let C = C.expect("hcontroller");
                let _ = SUNAdaptController_SetErrorBias(&C, 1.15);
                let _ = SUNAdaptController_SetParams_PI(&C, 0.8, -0.35);
                {
                    let mut m = ark_mem.borrow_mut();
                    let hadapt_mem = m.hadapt_mem.as_mut().expect("hadapt_mem");
                    hadapt_mem.safety = 0.993;
                    hadapt_mem.growth = 28.5;
                    hadapt_mem.etamxf = 0.3;
                    hadapt_mem.small_nef = SMALL_NEF;
                    hadapt_mem.etacf = ETACF;
                    hadapt_mem.pq = PQ;
                }
                {
                    let mut step_mem = arkStep_mem_mut(ark_mem);
                    step_mem.nlscoef = 0.25;
                    step_mem.crdown = 0.4;
                    step_mem.rdiv = 2.3;
                    step_mem.dgmax = 0.32;
                    step_mem.msbp = 31;
                }
            }
            _ => {}
        }
        /* NOTE (faithful to C): the ownership flag and the workspace
        accounting below sit INSIDE the ImEx branch upstream. */
        ark_mem
            .borrow_mut()
            .hadapt_mem
            .as_mut()
            .expect("hadapt_mem")
            .owncontroller = SUNTRUE;

        let hcontroller = {
            ark_mem
                .borrow()
                .hadapt_mem
                .as_ref()
                .expect("hadapt_mem")
                .hcontroller
                .clone()
        }
        .expect("hcontroller");
        retval = SUNAdaptController_Space(&hcontroller, &mut lenrw, &mut leniw);
        if retval == SUN_SUCCESS {
            let mut m = ark_mem.borrow_mut();
            m.liw += leniw;
            m.lrw += lenrw;
        }
    }
    ARK_SUCCESS
}

pub fn ARKStepSetOrder(arkode_mem: &ARKodeMem, ord: i32) -> i32 {
    ARKodeSetOrder(arkode_mem, ord)
}

pub fn ARKStepSetInterpolantType(arkode_mem: &ARKodeMem, itype: i32) -> i32 {
    ARKodeSetInterpolantType(arkode_mem, itype)
}

pub fn ARKStepSetInterpolantDegree(arkode_mem: &ARKodeMem, degree: i32) -> i32 {
    ARKodeSetInterpolantDegree(arkode_mem, degree)
}

pub fn ARKStepSetDenseOrder(arkode_mem: &ARKodeMem, dord: i32) -> i32 {
    ARKodeSetInterpolantDegree(arkode_mem, dord)
}

pub fn ARKStepSetNonlinearSolver(arkode_mem: &ARKodeMem, NLS: &SUNNonlinearSolver) -> i32 {
    ARKodeSetNonlinearSolver(arkode_mem, NLS)
}

pub fn ARKStepSetNlsRhsFn(arkode_mem: &ARKodeMem, nls_fi: Option<ARKRhsFn>) -> i32 {
    ARKodeSetNlsRhsFn(arkode_mem, nls_fi)
}

pub fn ARKStepSetLinear(arkode_mem: &ARKodeMem, timedepend: i32) -> i32 {
    ARKodeSetLinear(arkode_mem, timedepend)
}

pub fn ARKStepSetNonlinear(arkode_mem: &ARKodeMem) -> i32 {
    ARKodeSetNonlinear(arkode_mem)
}

pub fn ARKStepSetDeduceImplicitRhs(arkode_mem: &ARKodeMem, deduce: sunbooleantype) -> i32 {
    ARKodeSetDeduceImplicitRhs(arkode_mem, deduce)
}

pub fn ARKStepSetAdaptController(
    arkode_mem: &ARKodeMem,
    C: Option<&SUNAdaptController>,
) -> i32 {
    ARKodeSetAdaptController(arkode_mem, C)
}

pub fn ARKStepSetAdaptivityAdjustment(arkode_mem: &ARKodeMem, adjust: i32) -> i32 {
    ARKodeSetAdaptivityAdjustment(arkode_mem, adjust)
}

pub fn ARKStepSetCFLFraction(arkode_mem: &ARKodeMem, cfl_frac: sunrealtype) -> i32 {
    ARKodeSetCFLFraction(arkode_mem, cfl_frac)
}

pub fn ARKStepSetSafetyFactor(arkode_mem: &ARKodeMem, safety: sunrealtype) -> i32 {
    ARKodeSetSafetyFactor(arkode_mem, safety)
}

pub fn ARKStepSetErrorBias(arkode_mem: &ARKodeMem, bias: sunrealtype) -> i32 {
    ARKodeSetErrorBias(arkode_mem, bias)
}

pub fn ARKStepSetMaxGrowth(arkode_mem: &ARKodeMem, mx_growth: sunrealtype) -> i32 {
    ARKodeSetMaxGrowth(arkode_mem, mx_growth)
}

pub fn ARKStepSetMinReduction(arkode_mem: &ARKodeMem, eta_min: sunrealtype) -> i32 {
    ARKodeSetMinReduction(arkode_mem, eta_min)
}

pub fn ARKStepSetFixedStepBounds(
    arkode_mem: &ARKodeMem,
    lb: sunrealtype,
    ub: sunrealtype,
) -> i32 {
    ARKodeSetFixedStepBounds(arkode_mem, lb, ub)
}

pub fn ARKStepSetAdaptivityMethod(
    arkode_mem: &ARKodeMem,
    imethod: i32,
    idefault: i32,
    pq: i32,
    adapt_params: Option<&[sunrealtype; 3]>,
) -> i32 {
    arkSetAdaptivityMethod(arkode_mem, imethod, idefault, pq, adapt_params)
}

pub fn ARKStepSetAdaptivityFn(
    arkode_mem: &ARKodeMem,
    hfun: Option<ARKAdaptFn>,
    h_data: Option<Box<dyn Any>>,
) -> i32 {
    arkSetAdaptivityFn(arkode_mem, hfun, h_data)
}

pub fn ARKStepSetMaxFirstGrowth(arkode_mem: &ARKodeMem, etamx1: sunrealtype) -> i32 {
    ARKodeSetMaxFirstGrowth(arkode_mem, etamx1)
}

pub fn ARKStepSetMaxEFailGrowth(arkode_mem: &ARKodeMem, etamxf: sunrealtype) -> i32 {
    ARKodeSetMaxEFailGrowth(arkode_mem, etamxf)
}

pub fn ARKStepSetSmallNumEFails(arkode_mem: &ARKodeMem, small_nef: i32) -> i32 {
    ARKodeSetSmallNumEFails(arkode_mem, small_nef)
}

pub fn ARKStepSetMaxCFailGrowth(arkode_mem: &ARKodeMem, etacf: sunrealtype) -> i32 {
    ARKodeSetMaxCFailGrowth(arkode_mem, etacf)
}

pub fn ARKStepSetNonlinCRDown(arkode_mem: &ARKodeMem, crdown: sunrealtype) -> i32 {
    ARKodeSetNonlinCRDown(arkode_mem, crdown)
}

pub fn ARKStepSetNonlinRDiv(arkode_mem: &ARKodeMem, rdiv: sunrealtype) -> i32 {
    ARKodeSetNonlinRDiv(arkode_mem, rdiv)
}

pub fn ARKStepSetDeltaGammaMax(arkode_mem: &ARKodeMem, dgmax: sunrealtype) -> i32 {
    ARKodeSetDeltaGammaMax(arkode_mem, dgmax)
}

pub fn ARKStepSetLSetupFrequency(arkode_mem: &ARKodeMem, msbp: i32) -> i32 {
    ARKodeSetLSetupFrequency(arkode_mem, msbp)
}

pub fn ARKStepSetPredictorMethod(arkode_mem: &ARKodeMem, pred_method: i32) -> i32 {
    ARKodeSetPredictorMethod(arkode_mem, pred_method)
}

pub fn ARKStepSetStabilityFn(
    arkode_mem: &ARKodeMem,
    EStab: Option<ARKExpStabFn>,
    estab_data: Option<Box<dyn Any>>,
) -> i32 {
    ARKodeSetStabilityFn(arkode_mem, EStab, estab_data)
}

pub fn ARKStepSetMaxErrTestFails(arkode_mem: &ARKodeMem, maxnef: i32) -> i32 {
    ARKodeSetMaxErrTestFails(arkode_mem, maxnef)
}

pub fn ARKStepSetMaxNonlinIters(arkode_mem: &ARKodeMem, maxcor: i32) -> i32 {
    ARKodeSetMaxNonlinIters(arkode_mem, maxcor)
}

pub fn ARKStepSetMaxConvFails(arkode_mem: &ARKodeMem, maxncf: i32) -> i32 {
    ARKodeSetMaxConvFails(arkode_mem, maxncf)
}

pub fn ARKStepSetNonlinConvCoef(arkode_mem: &ARKodeMem, nlscoef: sunrealtype) -> i32 {
    ARKodeSetNonlinConvCoef(arkode_mem, nlscoef)
}

pub fn ARKStepSetConstraints(arkode_mem: &ARKodeMem, constraints: Option<&N_Vector>) -> i32 {
    ARKodeSetConstraints(arkode_mem, constraints)
}

pub fn ARKStepSetMaxNumSteps(arkode_mem: &ARKodeMem, mxsteps: i64) -> i32 {
    ARKodeSetMaxNumSteps(arkode_mem, mxsteps)
}

pub fn ARKStepSetMaxHnilWarns(arkode_mem: &ARKodeMem, mxhnil: i32) -> i32 {
    ARKodeSetMaxHnilWarns(arkode_mem, mxhnil)
}

pub fn ARKStepSetInitStep(arkode_mem: &ARKodeMem, hin: sunrealtype) -> i32 {
    ARKodeSetInitStep(arkode_mem, hin)
}

pub fn ARKStepSetMinStep(arkode_mem: &ARKodeMem, hmin: sunrealtype) -> i32 {
    ARKodeSetMinStep(arkode_mem, hmin)
}

pub fn ARKStepSetMaxStep(arkode_mem: &ARKodeMem, hmax: sunrealtype) -> i32 {
    ARKodeSetMaxStep(arkode_mem, hmax)
}

pub fn ARKStepSetInterpolateStopTime(arkode_mem: &ARKodeMem, interp: sunbooleantype) -> i32 {
    ARKodeSetInterpolateStopTime(arkode_mem, interp)
}

pub fn ARKStepSetStopTime(arkode_mem: &ARKodeMem, tstop: sunrealtype) -> i32 {
    ARKodeSetStopTime(arkode_mem, tstop)
}

pub fn ARKStepClearStopTime(arkode_mem: &ARKodeMem) -> i32 {
    ARKodeClearStopTime(arkode_mem)
}

pub fn ARKStepSetFixedStep(arkode_mem: &ARKodeMem, hfixed: sunrealtype) -> i32 {
    ARKodeSetFixedStep(arkode_mem, hfixed)
}

pub fn ARKStepSetMaxNumConstrFails(arkode_mem: &ARKodeMem, maxfails: i32) -> i32 {
    ARKodeSetMaxNumConstrFails(arkode_mem, maxfails)
}

pub fn ARKStepSetRootDirection(arkode_mem: &ARKodeMem, rootdir: &[i32]) -> i32 {
    ARKodeSetRootDirection(arkode_mem, rootdir)
}

pub fn ARKStepSetNoInactiveRootWarn(arkode_mem: &ARKodeMem) -> i32 {
    ARKodeSetNoInactiveRootWarn(arkode_mem)
}

pub fn ARKStepSetUserData(arkode_mem: &ARKodeMem, user_data: Option<Box<dyn Any>>) -> i32 {
    ARKodeSetUserData(arkode_mem, user_data)
}

pub fn ARKStepSetPostprocessStepFn(
    arkode_mem: &ARKodeMem,
    ProcessStep: Option<ARKPostProcessFn>,
) -> i32 {
    ARKodeSetPostprocessStepFn(arkode_mem, ProcessStep)
}

pub fn ARKStepSetPostprocessStageFn(
    arkode_mem: &ARKodeMem,
    ProcessStage: Option<ARKPostProcessFn>,
) -> i32 {
    ARKodeSetPostprocessStageFn(arkode_mem, ProcessStage)
}

pub fn ARKStepSetStagePredictFn(
    arkode_mem: &ARKodeMem,
    PredictStage: Option<ARKStagePredictFn>,
) -> i32 {
    ARKodeSetStagePredictFn(arkode_mem, PredictStage)
}

pub fn ARKStepSetJacFn(arkode_mem: &ARKodeMem, jac: Option<ARKLsJacFn>) -> i32 {
    ARKodeSetJacFn(arkode_mem, jac)
}

pub fn ARKStepSetMassFn(arkode_mem: &ARKodeMem, mass: Option<ARKLsMassFn>) -> i32 {
    ARKodeSetMassFn(arkode_mem, mass)
}

pub fn ARKStepSetJacEvalFrequency(arkode_mem: &ARKodeMem, msbj: i64) -> i32 {
    ARKodeSetJacEvalFrequency(arkode_mem, msbj)
}

pub fn ARKStepSetLinearSolutionScaling(arkode_mem: &ARKodeMem, onoff: sunbooleantype) -> i32 {
    ARKodeSetLinearSolutionScaling(arkode_mem, onoff)
}

pub fn ARKStepSetEpsLin(arkode_mem: &ARKodeMem, eplifac: sunrealtype) -> i32 {
    ARKodeSetEpsLin(arkode_mem, eplifac)
}

pub fn ARKStepSetMassEpsLin(arkode_mem: &ARKodeMem, eplifac: sunrealtype) -> i32 {
    ARKodeSetMassEpsLin(arkode_mem, eplifac)
}

pub fn ARKStepSetLSNormFactor(arkode_mem: &ARKodeMem, nrmfac: sunrealtype) -> i32 {
    ARKodeSetLSNormFactor(arkode_mem, nrmfac)
}

pub fn ARKStepSetMassLSNormFactor(arkode_mem: &ARKodeMem, nrmfac: sunrealtype) -> i32 {
    ARKodeSetMassLSNormFactor(arkode_mem, nrmfac)
}

pub fn ARKStepSetPreconditioner(
    arkode_mem: &ARKodeMem,
    psetup: Option<ARKLsPrecSetupFn>,
    psolve: Option<ARKLsPrecSolveFn>,
) -> i32 {
    ARKodeSetPreconditioner(arkode_mem, psetup, psolve)
}

pub fn ARKStepSetMassPreconditioner(
    arkode_mem: &ARKodeMem,
    psetup: Option<ARKLsMassPrecSetupFn>,
    psolve: Option<ARKLsMassPrecSolveFn>,
) -> i32 {
    ARKodeSetMassPreconditioner(arkode_mem, psetup, psolve)
}

pub fn ARKStepSetJacTimes(
    arkode_mem: &ARKodeMem,
    jtsetup: Option<ARKLsJacTimesSetupFn>,
    jtimes: Option<ARKLsJacTimesVecFn>,
) -> i32 {
    ARKodeSetJacTimes(arkode_mem, jtsetup, jtimes)
}

pub fn ARKStepSetJacTimesRhsFn(arkode_mem: &ARKodeMem, jtimesRhsFn: Option<ARKRhsFn>) -> i32 {
    ARKodeSetJacTimesRhsFn(arkode_mem, jtimesRhsFn)
}

pub fn ARKStepSetMassTimes(
    arkode_mem: &ARKodeMem,
    msetup: Option<ARKLsMassTimesSetupFn>,
    mtimes: Option<ARKLsMassTimesVecFn>,
    mtimes_data: Option<Box<dyn Any>>,
) -> i32 {
    ARKodeSetMassTimes(arkode_mem, msetup, mtimes, mtimes_data)
}

pub fn ARKStepSetLinSysFn(arkode_mem: &ARKodeMem, linsys: Option<ARKLsLinSysFn>) -> i32 {
    ARKodeSetLinSysFn(arkode_mem, linsys)
}

pub fn ARKStepEvolve(
    arkode_mem: &ARKodeMem,
    tout: sunrealtype,
    yout: &N_Vector,
    tret: &mut sunrealtype,
    itask: i32,
) -> i32 {
    ARKodeEvolve(arkode_mem, tout, yout, tret, itask)
}

pub fn ARKStepGetDky(arkode_mem: &ARKodeMem, t: sunrealtype, k: i32, dky: &N_Vector) -> i32 {
    ARKodeGetDky(arkode_mem, t, k, dky)
}

pub fn ARKStepComputeState(arkode_mem: &ARKodeMem, zcor: &N_Vector, z: &N_Vector) -> i32 {
    ARKodeComputeState(arkode_mem, zcor, z)
}

pub fn ARKStepGetNumExpSteps(arkode_mem: &ARKodeMem, nsteps: &mut i64) -> i32 {
    ARKodeGetNumExpSteps(arkode_mem, nsteps)
}

pub fn ARKStepGetNumAccSteps(arkode_mem: &ARKodeMem, nsteps: &mut i64) -> i32 {
    ARKodeGetNumAccSteps(arkode_mem, nsteps)
}

pub fn ARKStepGetNumStepAttempts(arkode_mem: &ARKodeMem, nstep_attempts: &mut i64) -> i32 {
    ARKodeGetNumStepAttempts(arkode_mem, nstep_attempts)
}

pub fn ARKStepGetNumLinSolvSetups(arkode_mem: &ARKodeMem, nlinsetups: &mut i64) -> i32 {
    ARKodeGetNumLinSolvSetups(arkode_mem, nlinsetups)
}

pub fn ARKStepGetNumErrTestFails(arkode_mem: &ARKodeMem, netfails: &mut i64) -> i32 {
    ARKodeGetNumErrTestFails(arkode_mem, netfails)
}

pub fn ARKStepGetEstLocalErrors(arkode_mem: &ARKodeMem, ele: &N_Vector) -> i32 {
    ARKodeGetEstLocalErrors(arkode_mem, ele)
}

pub fn ARKStepGetWorkSpace(arkode_mem: &ARKodeMem, lenrw: &mut i64, leniw: &mut i64) -> i32 {
    ARKodeGetWorkSpace(arkode_mem, lenrw, leniw)
}

pub fn ARKStepGetNumSteps(arkode_mem: &ARKodeMem, nsteps: &mut i64) -> i32 {
    ARKodeGetNumSteps(arkode_mem, nsteps)
}

pub fn ARKStepGetActualInitStep(arkode_mem: &ARKodeMem, hinused: &mut sunrealtype) -> i32 {
    ARKodeGetActualInitStep(arkode_mem, hinused)
}

pub fn ARKStepGetLastStep(arkode_mem: &ARKodeMem, hlast: &mut sunrealtype) -> i32 {
    ARKodeGetLastStep(arkode_mem, hlast)
}

pub fn ARKStepGetCurrentStep(arkode_mem: &ARKodeMem, hcur: &mut sunrealtype) -> i32 {
    ARKodeGetCurrentStep(arkode_mem, hcur)
}

pub fn ARKStepGetCurrentTime(arkode_mem: &ARKodeMem, tcur: &mut sunrealtype) -> i32 {
    ARKodeGetCurrentTime(arkode_mem, tcur)
}

pub fn ARKStepGetCurrentState(arkode_mem: &ARKodeMem, state: &mut Option<N_Vector>) -> i32 {
    ARKodeGetCurrentState(arkode_mem, state)
}

pub fn ARKStepGetCurrentGamma(arkode_mem: &ARKodeMem, gamma: &mut sunrealtype) -> i32 {
    ARKodeGetCurrentGamma(arkode_mem, gamma)
}

pub fn ARKStepGetCurrentMassMatrix(arkode_mem: &ARKodeMem, M: &mut Option<SUNMatrix>) -> i32 {
    ARKodeGetCurrentMassMatrix(arkode_mem, M)
}

pub fn ARKStepGetTolScaleFactor(arkode_mem: &ARKodeMem, tolsfact: &mut sunrealtype) -> i32 {
    ARKodeGetTolScaleFactor(arkode_mem, tolsfact)
}

pub fn ARKStepGetErrWeights(arkode_mem: &ARKodeMem, eweight: &N_Vector) -> i32 {
    ARKodeGetErrWeights(arkode_mem, eweight)
}

pub fn ARKStepGetResWeights(arkode_mem: &ARKodeMem, rweight: &N_Vector) -> i32 {
    ARKodeGetResWeights(arkode_mem, rweight)
}

pub fn ARKStepGetNumGEvals(arkode_mem: &ARKodeMem, ngevals: &mut i64) -> i32 {
    ARKodeGetNumGEvals(arkode_mem, ngevals)
}

pub fn ARKStepGetRootInfo(arkode_mem: &ARKodeMem, rootsfound: &mut [i32]) -> i32 {
    ARKodeGetRootInfo(arkode_mem, rootsfound)
}

pub fn ARKStepGetNumConstrFails(arkode_mem: &ARKodeMem, nconstrfails: &mut i64) -> i32 {
    ARKodeGetNumConstrFails(arkode_mem, nconstrfails)
}

pub fn ARKStepGetUserData(
    arkode_mem: &ARKodeMem,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    ARKodeGetUserData(arkode_mem, user_data)
}

pub fn ARKStepPrintAllStats(
    arkode_mem: &ARKodeMem,
    outfile: &SUNFile,
    fmt: SUNOutputFormat,
) -> i32 {
    ARKodePrintAllStats(arkode_mem, outfile, fmt)
}

/// C returns a `malloc`'d `char*`; the Rust port returns `String`.
pub fn ARKStepGetReturnFlagName(flag: i64) -> String {
    ARKodeGetReturnFlagName(flag)
}

pub fn ARKStepWriteParameters(arkode_mem: &ARKodeMem, fp: &SUNFile) -> i32 {
    ARKodeWriteParameters(arkode_mem, fp)
}

pub fn ARKStepWriteButcher(arkode_mem: &ARKodeMem, fp: &SUNFile) -> i32 {
    let ark_mem = arkode_mem;

    /* access ARKodeMem and ARKodeARKStepMem structures */
    let retval = arkStep_AccessARKODEStepMem(arkode_mem, "ARKStepWriteButcher");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* check that Butcher table is non-NULL (otherwise report error) */
    let (Be, Bi, stages, explicit, implicit) = {
        let step_mem = arkStep_mem_mut(ark_mem);
        (
            step_mem.Be.clone(),
            step_mem.Bi.clone(),
            step_mem.stages,
            step_mem.explicit,
            step_mem.implicit,
        )
    };
    if Be.is_none() && Bi.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "ARKStepWriteButcher",
            file!(),
            "Butcher table memory is NULL",
        );
        return ARK_MEM_NULL;
    }

    /* print Butcher tables to file */
    fp.write_str(&format!(
        "\nARKStep Butcher tables (stages = {}):\n",
        stages
    ));
    if explicit && Be.is_some() {
        fp.write_str("  Explicit Butcher table:\n");
        ARKodeButcherTable_Write(Be.as_ref(), fp);
    }
    fp.write_str("\n");
    if implicit && Bi.is_some() {
        fp.write_str("  Implicit Butcher table:\n");
        ARKodeButcherTable_Write(Bi.as_ref(), fp);
    }
    fp.write_str("\n");

    ARK_SUCCESS
}

pub fn ARKStepGetStepStats(
    arkode_mem: &ARKodeMem,
    nsteps: &mut i64,
    hinused: &mut sunrealtype,
    hlast: &mut sunrealtype,
    hcur: &mut sunrealtype,
    tcur: &mut sunrealtype,
) -> i32 {
    ARKodeGetStepStats(arkode_mem, nsteps, hinused, hlast, hcur, tcur)
}

#[allow(clippy::too_many_arguments)]
pub fn ARKStepGetNonlinearSystemData(
    arkode_mem: &ARKodeMem,
    tcur: &mut sunrealtype,
    zpred: &mut Option<N_Vector>,
    z: &mut Option<N_Vector>,
    Fi: &mut Option<N_Vector>,
    gamma: &mut sunrealtype,
    sdata: &mut Option<N_Vector>,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    ARKodeGetNonlinearSystemData(arkode_mem, tcur, zpred, z, Fi, gamma, sdata, user_data)
}

pub fn ARKStepGetNumNonlinSolvIters(arkode_mem: &ARKodeMem, nniters: &mut i64) -> i32 {
    ARKodeGetNumNonlinSolvIters(arkode_mem, nniters)
}

pub fn ARKStepGetNumNonlinSolvConvFails(arkode_mem: &ARKodeMem, nnfails: &mut i64) -> i32 {
    ARKodeGetNumNonlinSolvConvFails(arkode_mem, nnfails)
}

pub fn ARKStepGetNonlinSolvStats(
    arkode_mem: &ARKodeMem,
    nniters: &mut i64,
    nnfails: &mut i64,
) -> i32 {
    ARKodeGetNonlinSolvStats(arkode_mem, nniters, nnfails)
}

pub fn ARKStepGetNumStepSolveFails(arkode_mem: &ARKodeMem, nncfails: &mut i64) -> i32 {
    ARKodeGetNumStepSolveFails(arkode_mem, nncfails)
}

pub fn ARKStepGetJac(arkode_mem: &ARKodeMem, J: &mut Option<SUNMatrix>) -> i32 {
    ARKodeGetJac(arkode_mem, J)
}

pub fn ARKStepGetJacTime(arkode_mem: &ARKodeMem, t_J: &mut sunrealtype) -> i32 {
    ARKodeGetJacTime(arkode_mem, t_J)
}

pub fn ARKStepGetJacNumSteps(arkode_mem: &ARKodeMem, nst_J: &mut i64) -> i32 {
    ARKodeGetJacNumSteps(arkode_mem, nst_J)
}

pub fn ARKStepGetLinWorkSpace(
    arkode_mem: &ARKodeMem,
    lenrwLS: &mut i64,
    leniwLS: &mut i64,
) -> i32 {
    ARKodeGetLinWorkSpace(arkode_mem, lenrwLS, leniwLS)
}

pub fn ARKStepGetNumJacEvals(arkode_mem: &ARKodeMem, njevals: &mut i64) -> i32 {
    ARKodeGetNumJacEvals(arkode_mem, njevals)
}

pub fn ARKStepGetNumPrecEvals(arkode_mem: &ARKodeMem, npevals: &mut i64) -> i32 {
    ARKodeGetNumPrecEvals(arkode_mem, npevals)
}

pub fn ARKStepGetNumPrecSolves(arkode_mem: &ARKodeMem, npsolves: &mut i64) -> i32 {
    ARKodeGetNumPrecSolves(arkode_mem, npsolves)
}

pub fn ARKStepGetNumLinIters(arkode_mem: &ARKodeMem, nliters: &mut i64) -> i32 {
    ARKodeGetNumLinIters(arkode_mem, nliters)
}

pub fn ARKStepGetNumLinConvFails(arkode_mem: &ARKodeMem, nlcfails: &mut i64) -> i32 {
    ARKodeGetNumLinConvFails(arkode_mem, nlcfails)
}

pub fn ARKStepGetNumJTSetupEvals(arkode_mem: &ARKodeMem, njtsetups: &mut i64) -> i32 {
    ARKodeGetNumJTSetupEvals(arkode_mem, njtsetups)
}

pub fn ARKStepGetNumJtimesEvals(arkode_mem: &ARKodeMem, njvevals: &mut i64) -> i32 {
    ARKodeGetNumJtimesEvals(arkode_mem, njvevals)
}

pub fn ARKStepGetNumLinRhsEvals(arkode_mem: &ARKodeMem, nfevalsLS: &mut i64) -> i32 {
    ARKodeGetNumLinRhsEvals(arkode_mem, nfevalsLS)
}

pub fn ARKStepGetLastLinFlag(arkode_mem: &ARKodeMem, flag: &mut i64) -> i32 {
    ARKodeGetLastLinFlag(arkode_mem, flag)
}

pub fn ARKStepGetMassWorkSpace(
    arkode_mem: &ARKodeMem,
    lenrwMLS: &mut i64,
    leniwMLS: &mut i64,
) -> i32 {
    ARKodeGetMassWorkSpace(arkode_mem, lenrwMLS, leniwMLS)
}

pub fn ARKStepGetNumMassSetups(arkode_mem: &ARKodeMem, nmsetups: &mut i64) -> i32 {
    ARKodeGetNumMassSetups(arkode_mem, nmsetups)
}

pub fn ARKStepGetNumMassMultSetups(arkode_mem: &ARKodeMem, nmvsetups: &mut i64) -> i32 {
    ARKodeGetNumMassMultSetups(arkode_mem, nmvsetups)
}

pub fn ARKStepGetNumMassMult(arkode_mem: &ARKodeMem, nmvevals: &mut i64) -> i32 {
    ARKodeGetNumMassMult(arkode_mem, nmvevals)
}

pub fn ARKStepGetNumMassSolves(arkode_mem: &ARKodeMem, nmsolves: &mut i64) -> i32 {
    ARKodeGetNumMassSolves(arkode_mem, nmsolves)
}

pub fn ARKStepGetNumMassPrecEvals(arkode_mem: &ARKodeMem, nmpevals: &mut i64) -> i32 {
    ARKodeGetNumMassPrecEvals(arkode_mem, nmpevals)
}

pub fn ARKStepGetNumMassPrecSolves(arkode_mem: &ARKodeMem, nmpsolves: &mut i64) -> i32 {
    ARKodeGetNumMassPrecSolves(arkode_mem, nmpsolves)
}

pub fn ARKStepGetNumMassIters(arkode_mem: &ARKodeMem, nmiters: &mut i64) -> i32 {
    ARKodeGetNumMassIters(arkode_mem, nmiters)
}

pub fn ARKStepGetNumMassConvFails(arkode_mem: &ARKodeMem, nmcfails: &mut i64) -> i32 {
    ARKodeGetNumMassConvFails(arkode_mem, nmcfails)
}

pub fn ARKStepGetNumMTSetups(arkode_mem: &ARKodeMem, nmtsetups: &mut i64) -> i32 {
    ARKodeGetNumMTSetups(arkode_mem, nmtsetups)
}

pub fn ARKStepGetLastMassFlag(arkode_mem: &ARKodeMem, flag: &mut i64) -> i32 {
    ARKodeGetLastMassFlag(arkode_mem, flag)
}

/// C returns a `malloc`'d `char*`; the Rust port returns `String`.
pub fn ARKStepGetLinReturnFlagName(flag: i64) -> String {
    ARKodeGetLinReturnFlagName(flag)
}

pub fn ARKStepFree(arkode_mem: &mut Option<ARKodeMem>) {
    ARKodeFree(arkode_mem)
}

pub fn ARKStepPrintMem(arkode_mem: &ARKodeMem, outfile: &SUNFile) {
    ARKodePrintMem(arkode_mem, outfile)
}

pub fn ARKStepSetRelaxFn(
    arkode_mem: &ARKodeMem,
    rfn: Option<ARKRelaxFn>,
    rjac: Option<ARKRelaxJacFn>,
) -> i32 {
    ARKodeSetRelaxFn(arkode_mem, rfn, rjac)
}

pub fn ARKStepSetRelaxEtaFail(arkode_mem: &ARKodeMem, eta_rf: sunrealtype) -> i32 {
    ARKodeSetRelaxEtaFail(arkode_mem, eta_rf)
}

pub fn ARKStepSetRelaxLowerBound(arkode_mem: &ARKodeMem, lower: sunrealtype) -> i32 {
    ARKodeSetRelaxLowerBound(arkode_mem, lower)
}

pub fn ARKStepSetRelaxMaxFails(arkode_mem: &ARKodeMem, max_fails: i32) -> i32 {
    ARKodeSetRelaxMaxFails(arkode_mem, max_fails)
}

pub fn ARKStepSetRelaxMaxIters(arkode_mem: &ARKodeMem, max_iters: i32) -> i32 {
    ARKodeSetRelaxMaxIters(arkode_mem, max_iters)
}

pub fn ARKStepSetRelaxSolver(arkode_mem: &ARKodeMem, solver: ARKRelaxSolver) -> i32 {
    ARKodeSetRelaxSolver(arkode_mem, solver)
}

pub fn ARKStepSetRelaxResTol(arkode_mem: &ARKodeMem, res_tol: sunrealtype) -> i32 {
    ARKodeSetRelaxResTol(arkode_mem, res_tol)
}

pub fn ARKStepSetRelaxTol(
    arkode_mem: &ARKodeMem,
    rel_tol: sunrealtype,
    abs_tol: sunrealtype,
) -> i32 {
    ARKodeSetRelaxTol(arkode_mem, rel_tol, abs_tol)
}

pub fn ARKStepSetRelaxUpperBound(arkode_mem: &ARKodeMem, upper: sunrealtype) -> i32 {
    ARKodeSetRelaxUpperBound(arkode_mem, upper)
}

pub fn ARKStepGetNumRelaxFnEvals(arkode_mem: &ARKodeMem, r_evals: &mut i64) -> i32 {
    ARKodeGetNumRelaxFnEvals(arkode_mem, r_evals)
}

pub fn ARKStepGetNumRelaxJacEvals(arkode_mem: &ARKodeMem, J_evals: &mut i64) -> i32 {
    ARKodeGetNumRelaxJacEvals(arkode_mem, J_evals)
}

pub fn ARKStepGetNumRelaxFails(arkode_mem: &ARKodeMem, relax_fails: &mut i64) -> i32 {
    ARKodeGetNumRelaxFails(arkode_mem, relax_fails)
}

pub fn ARKStepGetNumRelaxBoundFails(arkode_mem: &ARKodeMem, fails: &mut i64) -> i32 {
    ARKodeGetNumRelaxBoundFails(arkode_mem, fails)
}

pub fn ARKStepGetNumRelaxSolveFails(arkode_mem: &ARKodeMem, fails: &mut i64) -> i32 {
    ARKodeGetNumRelaxSolveFails(arkode_mem, fails)
}

pub fn ARKStepGetNumRelaxSolveIters(arkode_mem: &ARKodeMem, iters: &mut i64) -> i32 {
    ARKodeGetNumRelaxSolveIters(arkode_mem, iters)
}

/*===============================================================
  EOF
  ===============================================================*/
