//! Port of `src/cvodes/cvodes_cli.c` (command-line control over optional
//! inputs to CVODES).
//!
//! The C key tables hold pointers to the public `CVodeSet*` setters,
//! which receive the raw `void* cvode_mem` forwarded by the
//! `sunCheckAndSet*Args` helpers. Here each table entry is a small
//! adapter matching `sundials_core::sundials_cli`'s setter fn types: it
//! downcasts the token (`Option<Box<dyn Any>>` holding a `CVodeMem`
//! clone) back to the handle and forwards to the real setter. Setters
//! whose C parameter is `sunbooleantype` (an `int` in C, fed directly
//! from `atoi`) convert with `arg != 0` — observably identical because
//! C only truth-tests the stored value.
//!
//! Relative to CVODE this file drops the fused-kernel key and adds the
//! sensitivity/quadrature keys plus five whole key groups for the
//! adjoint (`*_b`) setters: pair-of-int, int+real, int+long and
//! int+real+real.
//!
//! Cross-module note: the setters live in the sibling modules named
//! after their upstream C files — `cvodes_io.c`, `cvodes_ls.c`,
//! `cvodes_proj.c`, `cvodea_io.c`, `cvodes.c` and `cvodea.c`. If
//! `cvodes.c`/`cvodea.c` land as several fragment modules, only the four
//! `crate::cvodes::` / `crate::cvodea::` paths below (the `SStolerances`
//! family) need retargeting.

use std::any::Any;

use sundials_core::sundials_cli::*;
use sundials_core::sundials_types::*;

use crate::cvodes_impl::*;

/* -----------------------------------------------------------------
 * Adapter helpers: recover the CVodeMem handle from the CLI token
 * (C: the raw `void* cvode_mem` passed through sunCheckAndSet*Args).
 * A missing/mistyped token corresponds to C passing a garbage pointer
 * to the setter (UB) and maps to a deterministic panic.
 * ----------------------------------------------------------------- */

fn cliCVodeMem(mem: &mut Option<Box<dyn Any>>) -> CVodeMem {
    mem.as_mut()
        .and_then(|b| b.downcast_ref::<CVodeMem>())
        .cloned()
        .expect("cvode_mem token")
}

/* "int" setter adapters (table order below) */

fn cliCVodeSetMaxConvFails(mem: &mut Option<Box<dyn Any>>, arg: i32) -> i32 {
    let cv_mem = cliCVodeMem(mem);
    crate::cvodes_io::CVodeSetMaxConvFails(&cv_mem, arg)
}

fn cliCVodeSetMaxErrTestFails(mem: &mut Option<Box<dyn Any>>, arg: i32) -> i32 {
    let cv_mem = cliCVodeMem(mem);
    crate::cvodes_io::CVodeSetMaxErrTestFails(&cv_mem, arg)
}

fn cliCVodeSetMaxHnilWarns(mem: &mut Option<Box<dyn Any>>, arg: i32) -> i32 {
    let cv_mem = cliCVodeMem(mem);
    crate::cvodes_io::CVodeSetMaxHnilWarns(&cv_mem, arg)
}

fn cliCVodeSetMaxNonlinIters(mem: &mut Option<Box<dyn Any>>, arg: i32) -> i32 {
    let cv_mem = cliCVodeMem(mem);
    crate::cvodes_io::CVodeSetMaxNonlinIters(&cv_mem, arg)
}

fn cliCVodeSetMaxOrd(mem: &mut Option<Box<dyn Any>>, arg: i32) -> i32 {
    let cv_mem = cliCVodeMem(mem);
    crate::cvodes_io::CVodeSetMaxOrd(&cv_mem, arg)
}

fn cliCVodeSetStabLimDet(mem: &mut Option<Box<dyn Any>>, arg: i32) -> i32 {
    let cv_mem = cliCVodeMem(mem);
    /* C passes the int straight through as sunbooleantype */
    crate::cvodes_io::CVodeSetStabLimDet(&cv_mem, arg != 0)
}

fn cliCVodeSetInterpolateStopTime(mem: &mut Option<Box<dyn Any>>, arg: i32) -> i32 {
    let cv_mem = cliCVodeMem(mem);
    crate::cvodes_io::CVodeSetInterpolateStopTime(&cv_mem, arg != 0)
}

fn cliCVodeSetNumFailsEtaMaxErrFail(mem: &mut Option<Box<dyn Any>>, arg: i32) -> i32 {
    let cv_mem = cliCVodeMem(mem);
    crate::cvodes_io::CVodeSetNumFailsEtaMaxErrFail(&cv_mem, arg)
}

fn cliCVodeSetQuadErrCon(mem: &mut Option<Box<dyn Any>>, arg: i32) -> i32 {
    let cv_mem = cliCVodeMem(mem);
    crate::cvodes_io::CVodeSetQuadErrCon(&cv_mem, arg != 0)
}

fn cliCVodeSetSensErrCon(mem: &mut Option<Box<dyn Any>>, arg: i32) -> i32 {
    let cv_mem = cliCVodeMem(mem);
    crate::cvodes_io::CVodeSetSensErrCon(&cv_mem, arg != 0)
}

fn cliCVodeSetSensMaxNonlinIters(mem: &mut Option<Box<dyn Any>>, arg: i32) -> i32 {
    let cv_mem = cliCVodeMem(mem);
    crate::cvodes_io::CVodeSetSensMaxNonlinIters(&cv_mem, arg)
}

fn cliCVodeSetQuadSensErrCon(mem: &mut Option<Box<dyn Any>>, arg: i32) -> i32 {
    let cv_mem = cliCVodeMem(mem);
    crate::cvodes_io::CVodeSetQuadSensErrCon(&cv_mem, arg != 0)
}

fn cliCVodeSetLinearSolutionScaling(mem: &mut Option<Box<dyn Any>>, arg: i32) -> i32 {
    let cv_mem = cliCVodeMem(mem);
    crate::cvodes_ls::CVodeSetLinearSolutionScaling(&cv_mem, arg != 0)
}

fn cliCVodeSetProjErrEst(mem: &mut Option<Box<dyn Any>>, arg: i32) -> i32 {
    let cv_mem = cliCVodeMem(mem);
    crate::cvodes_proj::CVodeSetProjErrEst(&cv_mem, arg != 0)
}

fn cliCVodeSetMaxNumProjFails(mem: &mut Option<Box<dyn Any>>, arg: i32) -> i32 {
    let cv_mem = cliCVodeMem(mem);
    crate::cvodes_proj::CVodeSetMaxNumProjFails(&cv_mem, arg)
}

fn cliCVodeSetMaxNumConstraintFails(mem: &mut Option<Box<dyn Any>>, arg: i32) -> i32 {
    let cv_mem = cliCVodeMem(mem);
    crate::cvodes_io::CVodeSetMaxNumConstraintFails(&cv_mem, arg)
}

/* "long int" setter adapters (table order below) */

fn cliCVodeSetLSetupFrequency(mem: &mut Option<Box<dyn Any>>, arg: i64) -> i32 {
    let cv_mem = cliCVodeMem(mem);
    crate::cvodes_io::CVodeSetLSetupFrequency(&cv_mem, arg)
}

fn cliCVodeSetMaxNumSteps(mem: &mut Option<Box<dyn Any>>, arg: i64) -> i32 {
    let cv_mem = cliCVodeMem(mem);
    crate::cvodes_io::CVodeSetMaxNumSteps(&cv_mem, arg)
}

fn cliCVodeSetNumStepsEtaMaxEarlyStep(mem: &mut Option<Box<dyn Any>>, arg: i64) -> i32 {
    let cv_mem = cliCVodeMem(mem);
    crate::cvodes_io::CVodeSetNumStepsEtaMaxEarlyStep(&cv_mem, arg)
}

fn cliCVodeSetMonitorFrequency(mem: &mut Option<Box<dyn Any>>, arg: i64) -> i32 {
    let cv_mem = cliCVodeMem(mem);
    crate::cvodes_io::CVodeSetMonitorFrequency(&cv_mem, arg)
}

fn cliCVodeSetJacEvalFrequency(mem: &mut Option<Box<dyn Any>>, arg: i64) -> i32 {
    let cv_mem = cliCVodeMem(mem);
    crate::cvodes_ls::CVodeSetJacEvalFrequency(&cv_mem, arg)
}

fn cliCVodeSetProjFrequency(mem: &mut Option<Box<dyn Any>>, arg: i64) -> i32 {
    let cv_mem = cliCVodeMem(mem);
    crate::cvodes_proj::CVodeSetProjFrequency(&cv_mem, arg)
}

/* "sunrealtype" setter adapters (table order below) */

fn cliCVodeSetDeltaGammaMaxLSetup(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let cv_mem = cliCVodeMem(mem);
    crate::cvodes_io::CVodeSetDeltaGammaMaxLSetup(&cv_mem, arg)
}

fn cliCVodeSetInitStep(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let cv_mem = cliCVodeMem(mem);
    crate::cvodes_io::CVodeSetInitStep(&cv_mem, arg)
}

fn cliCVodeSetMaxStep(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let cv_mem = cliCVodeMem(mem);
    crate::cvodes_io::CVodeSetMaxStep(&cv_mem, arg)
}

fn cliCVodeSetMinStep(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let cv_mem = cliCVodeMem(mem);
    crate::cvodes_io::CVodeSetMinStep(&cv_mem, arg)
}

fn cliCVodeSetNonlinConvCoef(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let cv_mem = cliCVodeMem(mem);
    crate::cvodes_io::CVodeSetNonlinConvCoef(&cv_mem, arg)
}

fn cliCVodeSetStopTime(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let cv_mem = cliCVodeMem(mem);
    crate::cvodes_io::CVodeSetStopTime(&cv_mem, arg)
}

fn cliCVodeSetEtaMaxFirstStep(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let cv_mem = cliCVodeMem(mem);
    crate::cvodes_io::CVodeSetEtaMaxFirstStep(&cv_mem, arg)
}

fn cliCVodeSetEtaMaxEarlyStep(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let cv_mem = cliCVodeMem(mem);
    crate::cvodes_io::CVodeSetEtaMaxEarlyStep(&cv_mem, arg)
}

fn cliCVodeSetEtaMax(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let cv_mem = cliCVodeMem(mem);
    crate::cvodes_io::CVodeSetEtaMax(&cv_mem, arg)
}

fn cliCVodeSetEtaMin(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let cv_mem = cliCVodeMem(mem);
    crate::cvodes_io::CVodeSetEtaMin(&cv_mem, arg)
}

fn cliCVodeSetEtaMinErrFail(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let cv_mem = cliCVodeMem(mem);
    crate::cvodes_io::CVodeSetEtaMinErrFail(&cv_mem, arg)
}

fn cliCVodeSetEtaMaxErrFail(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let cv_mem = cliCVodeMem(mem);
    crate::cvodes_io::CVodeSetEtaMaxErrFail(&cv_mem, arg)
}

fn cliCVodeSetEtaConvFail(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let cv_mem = cliCVodeMem(mem);
    crate::cvodes_io::CVodeSetEtaConvFail(&cv_mem, arg)
}

fn cliCVodeSetDeltaGammaMaxBadJac(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let cv_mem = cliCVodeMem(mem);
    crate::cvodes_ls::CVodeSetDeltaGammaMaxBadJac(&cv_mem, arg)
}

fn cliCVodeSetEpsLin(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let cv_mem = cliCVodeMem(mem);
    crate::cvodes_ls::CVodeSetEpsLin(&cv_mem, arg)
}

fn cliCVodeSetLSNormFactor(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let cv_mem = cliCVodeMem(mem);
    crate::cvodes_ls::CVodeSetLSNormFactor(&cv_mem, arg)
}

fn cliCVodeSetEpsProj(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let cv_mem = cliCVodeMem(mem);
    crate::cvodes_proj::CVodeSetEpsProj(&cv_mem, arg)
}

fn cliCVodeSetProjFailEta(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let cv_mem = cliCVodeMem(mem);
    crate::cvodes_proj::CVodeSetProjFailEta(&cv_mem, arg)
}

/* pair-of-sunrealtype setter adapters (table order below) */

fn cliCVodeSetEtaFixedStepBounds(
    mem: &mut Option<Box<dyn Any>>,
    arg1: sunrealtype,
    arg2: sunrealtype,
) -> i32 {
    let cv_mem = cliCVodeMem(mem);
    crate::cvodes_io::CVodeSetEtaFixedStepBounds(&cv_mem, arg1, arg2)
}

fn cliCVodeSStolerances(
    mem: &mut Option<Box<dyn Any>>,
    arg1: sunrealtype,
    arg2: sunrealtype,
) -> i32 {
    let cv_mem = cliCVodeMem(mem);
    crate::cvodes::CVodeSStolerances(&cv_mem, arg1, arg2)
}

fn cliCVodeQuadSStolerances(
    mem: &mut Option<Box<dyn Any>>,
    arg1: sunrealtype,
    arg2: sunrealtype,
) -> i32 {
    let cv_mem = cliCVodeMem(mem);
    crate::cvodes::CVodeQuadSStolerances(&cv_mem, arg1, arg2)
}

/* pair-of-int setter adapters (table order below) */

fn cliCVodeSetMaxOrdB(mem: &mut Option<Box<dyn Any>>, arg1: i32, arg2: i32) -> i32 {
    let cv_mem = cliCVodeMem(mem);
    crate::cvodea_io::CVodeSetMaxOrdB(&cv_mem, arg1, arg2)
}

fn cliCVodeSetStabLimDetB(mem: &mut Option<Box<dyn Any>>, arg1: i32, arg2: i32) -> i32 {
    let cv_mem = cliCVodeMem(mem);
    crate::cvodea_io::CVodeSetStabLimDetB(&cv_mem, arg1, arg2 != 0)
}

fn cliCVodeSetQuadErrConB(mem: &mut Option<Box<dyn Any>>, arg1: i32, arg2: i32) -> i32 {
    let cv_mem = cliCVodeMem(mem);
    crate::cvodea_io::CVodeSetQuadErrConB(&cv_mem, arg1, arg2 != 0)
}

fn cliCVodeSetLinearSolutionScalingB(mem: &mut Option<Box<dyn Any>>, arg1: i32, arg2: i32) -> i32 {
    let cv_mem = cliCVodeMem(mem);
    crate::cvodes_ls::CVodeSetLinearSolutionScalingB(&cv_mem, arg1, arg2 != 0)
}

/* action setter adapters (table order below) */

fn cliCVodeClearStopTime(mem: &mut Option<Box<dyn Any>>) -> i32 {
    let cv_mem = cliCVodeMem(mem);
    crate::cvodes_io::CVodeClearStopTime(&cv_mem)
}

fn cliCVodeSetAdjNoSensi(mem: &mut Option<Box<dyn Any>>) -> i32 {
    let cv_mem = cliCVodeMem(mem);
    crate::cvodea_io::CVodeSetAdjNoSensi(&cv_mem)
}

fn cliCVodeSetNoInactiveRootWarn(mem: &mut Option<Box<dyn Any>>) -> i32 {
    let cv_mem = cliCVodeMem(mem);
    crate::cvodes_io::CVodeSetNoInactiveRootWarn(&cv_mem)
}

/* int+sunrealtype setter adapters (table order below) */

fn cliCVodeSetSensDQMethod(mem: &mut Option<Box<dyn Any>>, arg1: i32, arg2: sunrealtype) -> i32 {
    let cv_mem = cliCVodeMem(mem);
    crate::cvodes_io::CVodeSetSensDQMethod(&cv_mem, arg1, arg2)
}

fn cliCVodeSetInitStepB(mem: &mut Option<Box<dyn Any>>, arg1: i32, arg2: sunrealtype) -> i32 {
    let cv_mem = cliCVodeMem(mem);
    crate::cvodea_io::CVodeSetInitStepB(&cv_mem, arg1, arg2)
}

fn cliCVodeSetMinStepB(mem: &mut Option<Box<dyn Any>>, arg1: i32, arg2: sunrealtype) -> i32 {
    let cv_mem = cliCVodeMem(mem);
    crate::cvodea_io::CVodeSetMinStepB(&cv_mem, arg1, arg2)
}

fn cliCVodeSetMaxStepB(mem: &mut Option<Box<dyn Any>>, arg1: i32, arg2: sunrealtype) -> i32 {
    let cv_mem = cliCVodeMem(mem);
    crate::cvodea_io::CVodeSetMaxStepB(&cv_mem, arg1, arg2)
}

fn cliCVodeSetEpsLinB(mem: &mut Option<Box<dyn Any>>, arg1: i32, arg2: sunrealtype) -> i32 {
    let cv_mem = cliCVodeMem(mem);
    crate::cvodes_ls::CVodeSetEpsLinB(&cv_mem, arg1, arg2)
}

fn cliCVodeSetLSNormFactorB(mem: &mut Option<Box<dyn Any>>, arg1: i32, arg2: sunrealtype) -> i32 {
    let cv_mem = cliCVodeMem(mem);
    crate::cvodes_ls::CVodeSetLSNormFactorB(&cv_mem, arg1, arg2)
}

/* int+sunrealtype+sunrealtype setter adapters (table order below) */

fn cliCVodeSStolerancesB(
    mem: &mut Option<Box<dyn Any>>,
    arg1: i32,
    arg2: sunrealtype,
    arg3: sunrealtype,
) -> i32 {
    let cv_mem = cliCVodeMem(mem);
    crate::cvodea::CVodeSStolerancesB(&cv_mem, arg1, arg2, arg3)
}

fn cliCVodeQuadSStolerancesB(
    mem: &mut Option<Box<dyn Any>>,
    arg1: i32,
    arg2: sunrealtype,
    arg3: sunrealtype,
) -> i32 {
    let cv_mem = cliCVodeMem(mem);
    crate::cvodea::CVodeQuadSStolerancesB(&cv_mem, arg1, arg2, arg3)
}

/* int+long int setter adapters (table order below) */

fn cliCVodeSetMaxNumStepsB(mem: &mut Option<Box<dyn Any>>, arg1: i32, arg2: i64) -> i32 {
    let cv_mem = cliCVodeMem(mem);
    crate::cvodea_io::CVodeSetMaxNumStepsB(&cv_mem, arg1, arg2)
}

/*---------------------------------------------------------------
  CVodeSetOptions:

  Sets CVODE options using strings.
  ---------------------------------------------------------------*/

pub fn CVodeSetOptions(
    cvode_mem: &CVodeMem,
    cvid: Option<&str>,
    file_name: Option<&str>,
    argc: i32,
    argv: &[String],
) -> i32 {
    if let Some(file_name) = file_name {
        if !file_name.is_empty() {
            let retval = CV_ILL_INPUT;
            cvProcessError(
                Some(cvode_mem),
                retval,
                line!() as i32,
                "CVodeSetOptions",
                file!(),
                "file-based options are not currently supported.",
            );
            return retval;
        }
    }

    if argc > 0 {
        /* C also checks argv != NULL; slices are never null */
        let retval = cvSetFromCommandLine(cvode_mem, cvid, argc, argv);
        if retval != CV_SUCCESS {
            return retval;
        }
    }

    CV_SUCCESS
}

/* -----------------------------------------------------------------
 * Function to control CVODE options from the command line
 */

fn cvSetFromCommandLine(
    cvode_mem: &CVodeMem,
    cvid: Option<&str>,
    argc: i32,
    argv: &[String],
) -> i32 {
    /* NULL-mem check: handled by type system */
    let cv_mem = cvode_mem;

    /* Set lists of command-line arguments, and the corresponding set routines */
    static int_pairs: [sunKeyIntPair; 16] = [
        sunKeyIntPair { key: "max_conv_fails", set: cliCVodeSetMaxConvFails },
        sunKeyIntPair { key: "max_err_test_fails", set: cliCVodeSetMaxErrTestFails },
        sunKeyIntPair { key: "max_hnil_warns", set: cliCVodeSetMaxHnilWarns },
        sunKeyIntPair { key: "max_nonlin_iters", set: cliCVodeSetMaxNonlinIters },
        sunKeyIntPair { key: "max_order", set: cliCVodeSetMaxOrd },
        sunKeyIntPair { key: "stab_lim_det", set: cliCVodeSetStabLimDet },
        sunKeyIntPair { key: "interpolate_stop_time", set: cliCVodeSetInterpolateStopTime },
        sunKeyIntPair {
            key: "num_fails_eta_max_err_fail",
            set: cliCVodeSetNumFailsEtaMaxErrFail,
        },
        sunKeyIntPair { key: "quad_err_con", set: cliCVodeSetQuadErrCon },
        sunKeyIntPair { key: "sens_err_con", set: cliCVodeSetSensErrCon },
        sunKeyIntPair { key: "sens_max_nonlin_iters", set: cliCVodeSetSensMaxNonlinIters },
        sunKeyIntPair { key: "quad_sens_err_con", set: cliCVodeSetQuadSensErrCon },
        sunKeyIntPair { key: "linear_solution_scaling", set: cliCVodeSetLinearSolutionScaling },
        sunKeyIntPair { key: "proj_err_est", set: cliCVodeSetProjErrEst },
        sunKeyIntPair { key: "max_num_proj_fails", set: cliCVodeSetMaxNumProjFails },
        sunKeyIntPair {
            key: "max_num_constraint_fails",
            set: cliCVodeSetMaxNumConstraintFails,
        },
    ];
    let num_int_keys: i32 = int_pairs.len() as i32;

    static long_pairs: [sunKeyLongPair; 6] = [
        sunKeyLongPair { key: "lsetup_frequency", set: cliCVodeSetLSetupFrequency },
        sunKeyLongPair { key: "max_num_steps", set: cliCVodeSetMaxNumSteps },
        sunKeyLongPair {
            key: "num_steps_eta_max_early_step",
            set: cliCVodeSetNumStepsEtaMaxEarlyStep,
        },
        sunKeyLongPair { key: "monitor_frequency", set: cliCVodeSetMonitorFrequency },
        sunKeyLongPair { key: "jac_eval_frequency", set: cliCVodeSetJacEvalFrequency },
        sunKeyLongPair { key: "proj_frequency", set: cliCVodeSetProjFrequency },
    ];
    let num_long_keys: i32 = long_pairs.len() as i32;

    static real_pairs: [sunKeyRealPair; 18] = [
        sunKeyRealPair { key: "delta_gamma_max_lsetup", set: cliCVodeSetDeltaGammaMaxLSetup },
        sunKeyRealPair { key: "init_step", set: cliCVodeSetInitStep },
        sunKeyRealPair { key: "max_step", set: cliCVodeSetMaxStep },
        sunKeyRealPair { key: "min_step", set: cliCVodeSetMinStep },
        sunKeyRealPair { key: "nonlin_conv_coef", set: cliCVodeSetNonlinConvCoef },
        sunKeyRealPair { key: "stop_time", set: cliCVodeSetStopTime },
        sunKeyRealPair { key: "eta_max_first_step", set: cliCVodeSetEtaMaxFirstStep },
        sunKeyRealPair { key: "eta_max_early_step", set: cliCVodeSetEtaMaxEarlyStep },
        sunKeyRealPair { key: "eta_max", set: cliCVodeSetEtaMax },
        sunKeyRealPair { key: "eta_min", set: cliCVodeSetEtaMin },
        sunKeyRealPair { key: "eta_min_err_fail", set: cliCVodeSetEtaMinErrFail },
        sunKeyRealPair { key: "eta_max_err_fail", set: cliCVodeSetEtaMaxErrFail },
        sunKeyRealPair { key: "eta_conv_fail", set: cliCVodeSetEtaConvFail },
        sunKeyRealPair { key: "delta_gamma_max_bad_jac", set: cliCVodeSetDeltaGammaMaxBadJac },
        sunKeyRealPair { key: "eps_lin", set: cliCVodeSetEpsLin },
        sunKeyRealPair { key: "ls_norm_factor", set: cliCVodeSetLSNormFactor },
        sunKeyRealPair { key: "eps_proj", set: cliCVodeSetEpsProj },
        sunKeyRealPair { key: "proj_fail_eta", set: cliCVodeSetProjFailEta },
    ];
    let num_real_keys: i32 = real_pairs.len() as i32;

    static tworeal_pairs: [sunKeyTwoRealPair; 3] = [
        sunKeyTwoRealPair { key: "eta_fixed_step_bounds", set: cliCVodeSetEtaFixedStepBounds },
        sunKeyTwoRealPair { key: "scalar_tolerances", set: cliCVodeSStolerances },
        sunKeyTwoRealPair { key: "quad_scalar_tolerances", set: cliCVodeQuadSStolerances },
    ];
    let num_tworeal_keys: i32 = tworeal_pairs.len() as i32;

    static twoint_pairs: [sunKeyTwoIntPair; 4] = [
        sunKeyTwoIntPair { key: "max_order_b", set: cliCVodeSetMaxOrdB },
        sunKeyTwoIntPair { key: "stab_lim_det_b", set: cliCVodeSetStabLimDetB },
        sunKeyTwoIntPair { key: "quad_err_con_b", set: cliCVodeSetQuadErrConB },
        sunKeyTwoIntPair {
            key: "linear_solution_scaling_b",
            set: cliCVodeSetLinearSolutionScalingB,
        },
    ];
    let num_twoint_keys: i32 = twoint_pairs.len() as i32;

    static action_pairs: [sunKeyActionPair; 3] = [
        sunKeyActionPair { key: "clear_stop_time", set: cliCVodeClearStopTime },
        sunKeyActionPair { key: "adj_no_sensi", set: cliCVodeSetAdjNoSensi },
        sunKeyActionPair { key: "no_inactive_root_warn", set: cliCVodeSetNoInactiveRootWarn },
    ];
    let num_action_keys: i32 = action_pairs.len() as i32;

    static int_real_pairs: [sunKeyIntRealPair; 6] = [
        sunKeyIntRealPair { key: "sens_dq_method", set: cliCVodeSetSensDQMethod },
        sunKeyIntRealPair { key: "init_step_b", set: cliCVodeSetInitStepB },
        sunKeyIntRealPair { key: "min_step_b", set: cliCVodeSetMinStepB },
        sunKeyIntRealPair { key: "max_step_b", set: cliCVodeSetMaxStepB },
        sunKeyIntRealPair { key: "eps_lin_b", set: cliCVodeSetEpsLinB },
        sunKeyIntRealPair { key: "ls_norm_factor_b", set: cliCVodeSetLSNormFactorB },
    ];
    let num_int_real_keys: i32 = int_real_pairs.len() as i32;

    static int_real_real_pairs: [sunKeyIntRealRealPair; 2] = [
        sunKeyIntRealRealPair { key: "scalar_tolerances_b", set: cliCVodeSStolerancesB },
        sunKeyIntRealRealPair {
            key: "quad_scalar_tolerances_b",
            set: cliCVodeQuadSStolerancesB,
        },
    ];
    let num_int_real_real_keys: i32 = int_real_real_pairs.len() as i32;

    static int_long_pairs: [sunKeyIntLongPair; 1] =
        [sunKeyIntLongPair { key: "max_num_steps_b", set: cliCVodeSetMaxNumStepsB }];
    let num_int_long_keys: i32 = int_long_pairs.len() as i32;

    /* Prefix for options to set */
    let default_id = "cvodes";
    let mut offset: usize = default_id.len() + 1;
    if let Some(cvid) = cvid {
        if !cvid.is_empty() {
            offset = cvid.len() + 1;
        }
    }
    let mut prefix = String::with_capacity(offset + 1);
    match cvid {
        Some(cvid) if !cvid.is_empty() => prefix.push_str(cvid),
        _ => prefix.push_str(default_id),
    }
    prefix.push('.');

    /* the CLI helpers receive C's `void* cvode_mem` as a boxed handle clone */
    let mut mem: Option<Box<dyn Any>> = Some(Box::new(cvode_mem.clone()));

    let mut idx: i32 = 1;
    while idx < argc {
        'arg: {
            let mut j: i32 = 0;
            let mut arg_used: sunbooleantype = SUNFALSE;

            /* skip command-line arguments that do not begin with correct prefix */
            if !argv[idx as usize].starts_with(prefix.as_str()) {
                break 'arg;
            }

            /* check all "int" command-line options */
            let retval = sunCheckAndSetIntArgs(&mut mem, &mut idx, argv, offset, &int_pairs,
                                               num_int_keys, &mut arg_used, &mut j);
            if retval != CV_SUCCESS {
                cvProcessError(
                    Some(cv_mem),
                    retval,
                    line!() as i32,
                    "cvSetFromCommandLine",
                    file!(),
                    &format!("error setting key: {}", int_pairs[j as usize].key),
                );
                return retval;
            }
            if arg_used {
                break 'arg;
            }

            /* check all long int command-line options */
            let retval = sunCheckAndSetLongArgs(&mut mem, &mut idx, argv, offset, &long_pairs,
                                                num_long_keys, &mut arg_used, &mut j);
            if retval != CV_SUCCESS {
                cvProcessError(
                    Some(cv_mem),
                    retval,
                    line!() as i32,
                    "cvSetFromCommandLine",
                    file!(),
                    &format!("error setting key: {}", long_pairs[j as usize].key),
                );
                return retval;
            }
            if arg_used {
                break 'arg;
            }

            /* check all real command-line options */
            let retval = sunCheckAndSetRealArgs(&mut mem, &mut idx, argv, offset, &real_pairs,
                                                num_real_keys, &mut arg_used, &mut j);
            if retval != CV_SUCCESS {
                cvProcessError(
                    Some(cv_mem),
                    retval,
                    line!() as i32,
                    "cvSetFromCommandLine",
                    file!(),
                    &format!("error setting key: {}", real_pairs[j as usize].key),
                );
                return retval;
            }
            if arg_used {
                break 'arg;
            }

            /* check all pair-of-int command-line options */
            let retval = sunCheckAndSetTwoIntArgs(&mut mem, &mut idx, argv, offset, &twoint_pairs,
                                                  num_twoint_keys, &mut arg_used, &mut j);
            if retval != CV_SUCCESS {
                cvProcessError(
                    Some(cv_mem),
                    retval,
                    line!() as i32,
                    "cvSetFromCommandLine",
                    file!(),
                    &format!("error setting key: {}", twoint_pairs[j as usize].key),
                );
                return retval;
            }
            if arg_used {
                break 'arg;
            }

            /* check all pair-of-real command-line options */
            let retval = sunCheckAndSetTwoRealArgs(&mut mem, &mut idx, argv, offset,
                                                   &tworeal_pairs, num_tworeal_keys,
                                                   &mut arg_used, &mut j);
            if retval != CV_SUCCESS {
                cvProcessError(
                    Some(cv_mem),
                    retval,
                    line!() as i32,
                    "cvSetFromCommandLine",
                    file!(),
                    &format!("error setting key: {}", tworeal_pairs[j as usize].key),
                );
                return retval;
            }
            if arg_used {
                break 'arg;
            }

            /* check all action command-line options */
            let retval = sunCheckAndSetActionArgs(&mut mem, &mut idx, argv, offset, &action_pairs,
                                                  num_action_keys, &mut arg_used, &mut j);
            if retval != CV_SUCCESS {
                cvProcessError(
                    Some(cv_mem),
                    retval,
                    line!() as i32,
                    "cvSetFromCommandLine",
                    file!(),
                    &format!("error setting key: {}", action_pairs[j as usize].key),
                );
                return retval;
            }
            if arg_used {
                break 'arg;
            }

            /* check all int+real command-line options */
            let retval = sunCheckAndSetIntRealArgs(&mut mem, &mut idx, argv, offset,
                                                   &int_real_pairs, num_int_real_keys,
                                                   &mut arg_used, &mut j);
            if retval != CV_SUCCESS {
                cvProcessError(
                    Some(cv_mem),
                    retval,
                    line!() as i32,
                    "cvSetFromCommandLine",
                    file!(),
                    &format!("error setting key: {}", int_real_pairs[j as usize].key),
                );
                return retval;
            }
            if arg_used {
                break 'arg;
            }

            /* check all int+long command-line options */
            let retval = sunCheckAndSetIntLongArgs(&mut mem, &mut idx, argv, offset,
                                                   &int_long_pairs, num_int_long_keys,
                                                   &mut arg_used, &mut j);
            if retval != CV_SUCCESS {
                cvProcessError(
                    Some(cv_mem),
                    retval,
                    line!() as i32,
                    "cvSetFromCommandLine",
                    file!(),
                    &format!("error setting key: {}", int_long_pairs[j as usize].key),
                );
                return retval;
            }
            if arg_used {
                break 'arg;
            }

            /* check all int+real+real command-line options */
            let retval = sunCheckAndSetIntRealRealArgs(&mut mem, &mut idx, argv, offset,
                                                       &int_real_real_pairs,
                                                       num_int_real_real_keys, &mut arg_used,
                                                       &mut j);
            if retval != CV_SUCCESS {
                cvProcessError(
                    Some(cv_mem),
                    retval,
                    line!() as i32,
                    "cvSetFromCommandLine",
                    file!(),
                    &format!("error setting key: {}", int_real_real_pairs[j as usize].key),
                );
                return retval;
            }
            if arg_used {
                break 'arg;
            }

            /* warn for uninterpreted cvid.X arguments */
            cvProcessError(
                Some(cv_mem),
                CV_WARNING,
                line!() as i32,
                "cvSetFromCommandLine",
                file!(),
                &format!("WARNING: key {} was not handled\n", argv[idx as usize]),
            );
        }
        idx += 1;
    }
    CV_SUCCESS
}

/*===============================================================
  EOF
  ===============================================================*/
