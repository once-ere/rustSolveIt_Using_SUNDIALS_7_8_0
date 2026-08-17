//! Port of `src/arkode/arkode_cli.c` (command-line control over optional
//! inputs to ARKODE).
//!
//! Same shape as `cvode_rs::cvode_cli`: the C key tables hold pointers to
//! the public `ARKodeSet*` setters, which receive the raw
//! `void* arkode_mem` forwarded by the `sunCheckAndSet*Args` helpers.
//! Here each table entry is a small adapter matching
//! `sundials_core::sundials_cli`'s setter fn types: it downcasts the token
//! (`Option<Box<dyn Any>>` holding an `ARKodeMem` clone) back to the handle
//! and forwards to the real setter. Setters whose C parameter is
//! `sunbooleantype` (an `int` in C, fed straight from `atoi`) convert with
//! `arg != 0` — observably identical because C only truth-tests the stored
//! value.
//!
//! The key tables keep the C order byte-for-byte; the C `continue`
//! statements inside the argument loop become `break 'arg` out of a
//! labeled block so the loop increment still runs, exactly as `for (idx…;
//! idx++)` does.

use std::any::Any;

use sundials_core::sundials_cli::*;
use sundials_core::sundials_types::*;
use sundials_core::sundials_utils::SUNFile;

use crate::arkode_impl::*;

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

/* "int" setter adapters (table order below) */

fn cliARKodeSetOrder(mem: &mut Option<Box<dyn Any>>, arg: i32) -> i32 {
    let ark_mem = cliARKodeMem(mem);
    crate::arkode_io::ARKodeSetOrder(&ark_mem, arg)
}

fn cliARKodeSetInterpolantDegree(mem: &mut Option<Box<dyn Any>>, arg: i32) -> i32 {
    let ark_mem = cliARKodeMem(mem);
    crate::arkode_io::ARKodeSetInterpolantDegree(&ark_mem, arg)
}

fn cliARKodeSetLinear(mem: &mut Option<Box<dyn Any>>, arg: i32) -> i32 {
    let ark_mem = cliARKodeMem(mem);
    crate::arkode_io::ARKodeSetLinear(&ark_mem, arg)
}

fn cliARKodeSetAutonomous(mem: &mut Option<Box<dyn Any>>, arg: i32) -> i32 {
    let ark_mem = cliARKodeMem(mem);
    /* C passes the int straight through as sunbooleantype */
    crate::arkode_io::ARKodeSetAutonomous(&ark_mem, arg != 0)
}

fn cliARKodeSetDeduceImplicitRhs(mem: &mut Option<Box<dyn Any>>, arg: i32) -> i32 {
    let ark_mem = cliARKodeMem(mem);
    crate::arkode_io::ARKodeSetDeduceImplicitRhs(&ark_mem, arg != 0)
}

fn cliARKodeSetLSetupFrequency(mem: &mut Option<Box<dyn Any>>, arg: i32) -> i32 {
    let ark_mem = cliARKodeMem(mem);
    crate::arkode_io::ARKodeSetLSetupFrequency(&ark_mem, arg)
}

fn cliARKodeSetPredictorMethod(mem: &mut Option<Box<dyn Any>>, arg: i32) -> i32 {
    let ark_mem = cliARKodeMem(mem);
    crate::arkode_io::ARKodeSetPredictorMethod(&ark_mem, arg)
}

fn cliARKodeSetMaxNonlinIters(mem: &mut Option<Box<dyn Any>>, arg: i32) -> i32 {
    let ark_mem = cliARKodeMem(mem);
    crate::arkode_io::ARKodeSetMaxNonlinIters(&ark_mem, arg)
}

fn cliARKodeSetMaxHnilWarns(mem: &mut Option<Box<dyn Any>>, arg: i32) -> i32 {
    let ark_mem = cliARKodeMem(mem);
    crate::arkode_io::ARKodeSetMaxHnilWarns(&ark_mem, arg)
}

fn cliARKodeSetInterpolateStopTime(mem: &mut Option<Box<dyn Any>>, arg: i32) -> i32 {
    let ark_mem = cliARKodeMem(mem);
    crate::arkode_io::ARKodeSetInterpolateStopTime(&ark_mem, arg != 0)
}

fn cliARKodeSetMaxNumConstrFails(mem: &mut Option<Box<dyn Any>>, arg: i32) -> i32 {
    let ark_mem = cliARKodeMem(mem);
    crate::arkode_io::ARKodeSetMaxNumConstrFails(&ark_mem, arg)
}

fn cliARKodeSetAdaptivityAdjustment(mem: &mut Option<Box<dyn Any>>, arg: i32) -> i32 {
    let ark_mem = cliARKodeMem(mem);
    crate::arkode_io::ARKodeSetAdaptivityAdjustment(&ark_mem, arg)
}

fn cliARKodeSetSmallNumEFails(mem: &mut Option<Box<dyn Any>>, arg: i32) -> i32 {
    let ark_mem = cliARKodeMem(mem);
    crate::arkode_io::ARKodeSetSmallNumEFails(&ark_mem, arg)
}

fn cliARKodeSetMaxErrTestFails(mem: &mut Option<Box<dyn Any>>, arg: i32) -> i32 {
    let ark_mem = cliARKodeMem(mem);
    crate::arkode_io::ARKodeSetMaxErrTestFails(&ark_mem, arg)
}

fn cliARKodeSetMaxConvFails(mem: &mut Option<Box<dyn Any>>, arg: i32) -> i32 {
    let ark_mem = cliARKodeMem(mem);
    crate::arkode_io::ARKodeSetMaxConvFails(&ark_mem, arg)
}

fn cliARKodeSetLinearSolutionScaling(mem: &mut Option<Box<dyn Any>>, arg: i32) -> i32 {
    let ark_mem = cliARKodeMem(mem);
    crate::arkode_ls::ARKodeSetLinearSolutionScaling(&ark_mem, arg != 0)
}

fn cliARKodeSetUseCompensatedSums(mem: &mut Option<Box<dyn Any>>, arg: i32) -> i32 {
    let ark_mem = cliARKodeMem(mem);
    crate::arkode_io::ARKodeSetUseCompensatedSums(&ark_mem, arg != 0)
}

/* "long int" setter adapters (table order below) */

fn cliARKodeSetMaxNumSteps(mem: &mut Option<Box<dyn Any>>, arg: i64) -> i32 {
    let ark_mem = cliARKodeMem(mem);
    crate::arkode_io::ARKodeSetMaxNumSteps(&ark_mem, arg)
}

fn cliARKodeSetJacEvalFrequency(mem: &mut Option<Box<dyn Any>>, arg: i64) -> i32 {
    let ark_mem = cliARKodeMem(mem);
    crate::arkode_ls::ARKodeSetJacEvalFrequency(&ark_mem, arg)
}

/* "sunrealtype" setter adapters (table order below) */

fn cliARKodeSetNonlinCRDown(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let ark_mem = cliARKodeMem(mem);
    crate::arkode_io::ARKodeSetNonlinCRDown(&ark_mem, arg)
}

fn cliARKodeSetNonlinRDiv(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let ark_mem = cliARKodeMem(mem);
    crate::arkode_io::ARKodeSetNonlinRDiv(&ark_mem, arg)
}

fn cliARKodeSetDeltaGammaMax(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let ark_mem = cliARKodeMem(mem);
    crate::arkode_io::ARKodeSetDeltaGammaMax(&ark_mem, arg)
}

fn cliARKodeSetNonlinConvCoef(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let ark_mem = cliARKodeMem(mem);
    crate::arkode_io::ARKodeSetNonlinConvCoef(&ark_mem, arg)
}

fn cliARKodeSetInitStep(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let ark_mem = cliARKodeMem(mem);
    crate::arkode_io::ARKodeSetInitStep(&ark_mem, arg)
}

fn cliARKodeSetMinStep(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let ark_mem = cliARKodeMem(mem);
    crate::arkode_io::ARKodeSetMinStep(&ark_mem, arg)
}

fn cliARKodeSetMaxStep(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let ark_mem = cliARKodeMem(mem);
    crate::arkode_io::ARKodeSetMaxStep(&ark_mem, arg)
}

fn cliARKodeSetStopTime(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let ark_mem = cliARKodeMem(mem);
    crate::arkode_io::ARKodeSetStopTime(&ark_mem, arg)
}

fn cliARKodeSetFixedStep(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let ark_mem = cliARKodeMem(mem);
    crate::arkode_io::ARKodeSetFixedStep(&ark_mem, arg)
}

fn cliARKodeSetStepDirection(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let ark_mem = cliARKodeMem(mem);
    crate::arkode_io::ARKodeSetStepDirection(&ark_mem, arg)
}

fn cliARKodeSetCFLFraction(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let ark_mem = cliARKodeMem(mem);
    crate::arkode_io::ARKodeSetCFLFraction(&ark_mem, arg)
}

fn cliARKodeSetSafetyFactor(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let ark_mem = cliARKodeMem(mem);
    crate::arkode_io::ARKodeSetSafetyFactor(&ark_mem, arg)
}

fn cliARKodeSetErrorBias(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let ark_mem = cliARKodeMem(mem);
    crate::arkode_io::ARKodeSetErrorBias(&ark_mem, arg)
}

fn cliARKodeSetMaxGrowth(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let ark_mem = cliARKodeMem(mem);
    crate::arkode_io::ARKodeSetMaxGrowth(&ark_mem, arg)
}

fn cliARKodeSetMinReduction(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let ark_mem = cliARKodeMem(mem);
    crate::arkode_io::ARKodeSetMinReduction(&ark_mem, arg)
}

fn cliARKodeSetMaxFirstGrowth(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let ark_mem = cliARKodeMem(mem);
    crate::arkode_io::ARKodeSetMaxFirstGrowth(&ark_mem, arg)
}

fn cliARKodeSetMaxEFailGrowth(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let ark_mem = cliARKodeMem(mem);
    crate::arkode_io::ARKodeSetMaxEFailGrowth(&ark_mem, arg)
}

fn cliARKodeSetMaxCFailGrowth(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let ark_mem = cliARKodeMem(mem);
    crate::arkode_io::ARKodeSetMaxCFailGrowth(&ark_mem, arg)
}

fn cliARKodeSetEpsLin(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let ark_mem = cliARKodeMem(mem);
    crate::arkode_ls::ARKodeSetEpsLin(&ark_mem, arg)
}

fn cliARKodeSetMassEpsLin(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let ark_mem = cliARKodeMem(mem);
    crate::arkode_ls::ARKodeSetMassEpsLin(&ark_mem, arg)
}

fn cliARKodeSetLSNormFactor(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let ark_mem = cliARKodeMem(mem);
    crate::arkode_ls::ARKodeSetLSNormFactor(&ark_mem, arg)
}

fn cliARKodeSetMassLSNormFactor(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let ark_mem = cliARKodeMem(mem);
    crate::arkode_ls::ARKodeSetMassLSNormFactor(&ark_mem, arg)
}

/* pair-of-sunrealtype setter adapters (table order below) */

fn cliARKodeSStolerances(
    mem: &mut Option<Box<dyn Any>>,
    arg1: sunrealtype,
    arg2: sunrealtype,
) -> i32 {
    let ark_mem = cliARKodeMem(mem);
    crate::arkode::ARKodeSStolerances(&ark_mem, arg1, arg2)
}

fn cliARKodeSetFixedStepBounds(
    mem: &mut Option<Box<dyn Any>>,
    arg1: sunrealtype,
    arg2: sunrealtype,
) -> i32 {
    let ark_mem = cliARKodeMem(mem);
    crate::arkode_io::ARKodeSetFixedStepBounds(&ark_mem, arg1, arg2)
}

/* action setter adapters (table order below) */

fn cliARKodeSetNonlinear(mem: &mut Option<Box<dyn Any>>) -> i32 {
    let ark_mem = cliARKodeMem(mem);
    crate::arkode_io::ARKodeSetNonlinear(&ark_mem)
}

fn cliARKodeClearStopTime(mem: &mut Option<Box<dyn Any>>) -> i32 {
    let ark_mem = cliARKodeMem(mem);
    crate::arkode_io::ARKodeClearStopTime(&ark_mem)
}

fn cliARKodeSetNoInactiveRootWarn(mem: &mut Option<Box<dyn Any>>) -> i32 {
    let ark_mem = cliARKodeMem(mem);
    crate::arkode_io::ARKodeSetNoInactiveRootWarn(&ark_mem)
}

fn cliARKodeResetAccumulatedError(mem: &mut Option<Box<dyn Any>>) -> i32 {
    let ark_mem = cliARKodeMem(mem);
    crate::arkode_io::ARKodeResetAccumulatedError(&ark_mem)
}

/*---------------------------------------------------------------
  ARKodeSetOptions:

  Sets ARKODE options using strings.
  ---------------------------------------------------------------*/

pub fn ARKodeSetOptions(
    arkode_mem: &ARKodeMem,
    arkid: Option<&str>,
    file_name: Option<&str>,
    argc: i32,
    argv: &[String],
) -> i32 {
    if let Some(file_name) = file_name {
        if !file_name.is_empty() {
            let retval = ARK_ILL_INPUT;
            arkProcessError(
                Some(arkode_mem),
                retval,
                line!() as i32,
                "ARKodeSetOptions",
                file!(),
                "file-based options are not currently supported.",
            );
            return retval;
        }
    }

    if argc > 0 {
        /* C also checks argv != NULL; slices are never null */
        let retval = arkSetFromCommandLine(arkode_mem, arkid, argc, argv);
        if retval != ARK_SUCCESS {
            return retval;
        }
    }

    ARK_SUCCESS
}

/* -----------------------------------------------------------------
 * Function to control ARKODE options from the command line
 */

fn arkSetFromCommandLine(
    arkode_mem: &ARKodeMem,
    arkid: Option<&str>,
    argc: i32,
    argv: &[String],
) -> i32 {
    /* NULL-mem check: handled by type system */
    let ark_mem = arkode_mem;

    /* Set lists of command-line arguments, and the corresponding set routines */
    static int_pairs: [sunKeyIntPair; 17] = [
        sunKeyIntPair { key: "order", set: cliARKodeSetOrder },
        sunKeyIntPair { key: "interpolant_degree", set: cliARKodeSetInterpolantDegree },
        sunKeyIntPair { key: "linear", set: cliARKodeSetLinear },
        sunKeyIntPair { key: "autonomous", set: cliARKodeSetAutonomous },
        sunKeyIntPair { key: "deduce_implicit_rhs", set: cliARKodeSetDeduceImplicitRhs },
        sunKeyIntPair { key: "lsetup_frequency", set: cliARKodeSetLSetupFrequency },
        sunKeyIntPair { key: "predictor_method", set: cliARKodeSetPredictorMethod },
        sunKeyIntPair { key: "max_nonlin_iters", set: cliARKodeSetMaxNonlinIters },
        sunKeyIntPair { key: "max_hnil_warns", set: cliARKodeSetMaxHnilWarns },
        sunKeyIntPair { key: "interpolate_stop_time", set: cliARKodeSetInterpolateStopTime },
        sunKeyIntPair { key: "max_num_constr_fails", set: cliARKodeSetMaxNumConstrFails },
        sunKeyIntPair { key: "adaptivity_adjustment", set: cliARKodeSetAdaptivityAdjustment },
        sunKeyIntPair { key: "small_num_efails", set: cliARKodeSetSmallNumEFails },
        sunKeyIntPair { key: "max_err_test_fails", set: cliARKodeSetMaxErrTestFails },
        sunKeyIntPair { key: "max_conv_fails", set: cliARKodeSetMaxConvFails },
        sunKeyIntPair { key: "linear_solution_scaling", set: cliARKodeSetLinearSolutionScaling },
        sunKeyIntPair { key: "use_compensated_sums", set: cliARKodeSetUseCompensatedSums },
    ];
    let num_int_keys: i32 = int_pairs.len() as i32;

    static long_pairs: [sunKeyLongPair; 2] = [
        sunKeyLongPair { key: "max_num_steps", set: cliARKodeSetMaxNumSteps },
        sunKeyLongPair { key: "jac_eval_frequency", set: cliARKodeSetJacEvalFrequency },
    ];
    let num_long_keys: i32 = long_pairs.len() as i32;

    static real_pairs: [sunKeyRealPair; 22] = [
        sunKeyRealPair { key: "nonlin_crdown", set: cliARKodeSetNonlinCRDown },
        sunKeyRealPair { key: "nonlin_rdiv", set: cliARKodeSetNonlinRDiv },
        sunKeyRealPair { key: "delta_gamma_max", set: cliARKodeSetDeltaGammaMax },
        sunKeyRealPair { key: "nonlin_conv_coef", set: cliARKodeSetNonlinConvCoef },
        sunKeyRealPair { key: "init_step", set: cliARKodeSetInitStep },
        sunKeyRealPair { key: "min_step", set: cliARKodeSetMinStep },
        sunKeyRealPair { key: "max_step", set: cliARKodeSetMaxStep },
        sunKeyRealPair { key: "stop_time", set: cliARKodeSetStopTime },
        sunKeyRealPair { key: "fixed_step", set: cliARKodeSetFixedStep },
        sunKeyRealPair { key: "step_direction", set: cliARKodeSetStepDirection },
        sunKeyRealPair { key: "cfl_fraction", set: cliARKodeSetCFLFraction },
        sunKeyRealPair { key: "safety_factor", set: cliARKodeSetSafetyFactor },
        sunKeyRealPair { key: "error_bias", set: cliARKodeSetErrorBias },
        sunKeyRealPair { key: "max_growth", set: cliARKodeSetMaxGrowth },
        sunKeyRealPair { key: "min_reduction", set: cliARKodeSetMinReduction },
        sunKeyRealPair { key: "max_first_growth", set: cliARKodeSetMaxFirstGrowth },
        sunKeyRealPair { key: "max_efail_growth", set: cliARKodeSetMaxEFailGrowth },
        sunKeyRealPair { key: "max_cfail_growth", set: cliARKodeSetMaxCFailGrowth },
        sunKeyRealPair { key: "eps_lin", set: cliARKodeSetEpsLin },
        sunKeyRealPair { key: "mass_eps_lin", set: cliARKodeSetMassEpsLin },
        sunKeyRealPair { key: "ls_norm_factor", set: cliARKodeSetLSNormFactor },
        sunKeyRealPair { key: "mass_ls_norm_factor", set: cliARKodeSetMassLSNormFactor },
    ];
    let num_real_keys: i32 = real_pairs.len() as i32;

    static tworeal_pairs: [sunKeyTwoRealPair; 2] = [
        sunKeyTwoRealPair { key: "scalar_tolerances", set: cliARKodeSStolerances },
        sunKeyTwoRealPair { key: "fixed_step_bounds", set: cliARKodeSetFixedStepBounds },
    ];
    let num_tworeal_keys: i32 = tworeal_pairs.len() as i32;

    static action_pairs: [sunKeyActionPair; 4] = [
        sunKeyActionPair { key: "nonlinear", set: cliARKodeSetNonlinear },
        sunKeyActionPair { key: "clear_stop_time", set: cliARKodeClearStopTime },
        sunKeyActionPair { key: "no_inactive_root_warn", set: cliARKodeSetNoInactiveRootWarn },
        sunKeyActionPair { key: "reset_accumulated_error", set: cliARKodeResetAccumulatedError },
    ];
    let num_action_keys: i32 = action_pairs.len() as i32;

    /* Prefix for options to set */
    let default_id = "arkode";
    let mut offset: usize = default_id.len() + 1;
    if let Some(arkid) = arkid {
        if !arkid.is_empty() {
            offset = arkid.len() + 1;
        }
    }
    let mut prefix = String::with_capacity(offset + 1);
    match arkid {
        Some(arkid) if !arkid.is_empty() => prefix.push_str(arkid),
        _ => prefix.push_str(default_id),
    }
    prefix.push('.');

    /* the CLI helpers receive C's `void* arkode_mem` as a boxed handle clone */
    let mut mem: Option<Box<dyn Any>> = Some(Box::new(arkode_mem.clone()));

    let mut write_parameters: sunbooleantype = SUNFALSE;
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
            if retval != ARK_SUCCESS {
                arkProcessError(
                    Some(ark_mem),
                    retval,
                    line!() as i32,
                    "arkSetFromCommandLine",
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
            if retval != ARK_SUCCESS {
                arkProcessError(
                    Some(ark_mem),
                    retval,
                    line!() as i32,
                    "arkSetFromCommandLine",
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
            if retval != ARK_SUCCESS {
                arkProcessError(
                    Some(ark_mem),
                    retval,
                    line!() as i32,
                    "arkSetFromCommandLine",
                    file!(),
                    &format!("error setting key: {}", real_pairs[j as usize].key),
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
            if retval != ARK_SUCCESS {
                arkProcessError(
                    Some(ark_mem),
                    retval,
                    line!() as i32,
                    "arkSetFromCommandLine",
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
            if retval != ARK_SUCCESS {
                arkProcessError(
                    Some(ark_mem),
                    retval,
                    line!() as i32,
                    "arkSetFromCommandLine",
                    file!(),
                    &format!("error setting key: {}", action_pairs[j as usize].key),
                );
                return retval;
            }
            if arg_used {
                break 'arg;
            }

            /*** handle all remaining command-line options ***/

            if argv[idx as usize][offset..] == *"interpolant_type" {
                idx += 1;
                let mut retval = ARK_ILL_INPUT;
                if argv[idx as usize] == *"ARK_INTERP_HERMITE" {
                    retval =
                        crate::arkode_io::ARKodeSetInterpolantType(arkode_mem, ARK_INTERP_HERMITE);
                } else if argv[idx as usize] == *"ARK_INTERP_LAGRANGE" {
                    retval =
                        crate::arkode_io::ARKodeSetInterpolantType(arkode_mem, ARK_INTERP_LAGRANGE);
                } else if argv[idx as usize] == *"ARK_INTERP_NONE" {
                    retval = crate::arkode_io::ARKodeSetInterpolantType(arkode_mem, ARK_INTERP_NONE);
                }
                if retval != ARK_SUCCESS {
                    arkProcessError(
                        Some(ark_mem),
                        retval,
                        line!() as i32,
                        "arkSetFromCommandLine",
                        file!(),
                        &format!(
                            "error setting key: {} {}",
                            argv[(idx - 1) as usize],
                            argv[idx as usize]
                        ),
                    );
                    return retval;
                }
                /* C also sets `arg_used = SUNTRUE` here; the value is never
                read again before the loop body ends (dead store) */
                break 'arg;
            }

            if argv[idx as usize][offset..] == *"accumulated_error_type" {
                idx += 1;
                let mut retval = ARK_ILL_INPUT;
                if argv[idx as usize] == *"ARK_ACCUMERROR_NONE" {
                    retval = crate::arkode_io::ARKodeSetAccumulatedErrorType(
                        arkode_mem,
                        ARK_ACCUMERROR_NONE,
                    );
                } else if argv[idx as usize] == *"ARK_ACCUMERROR_MAX" {
                    retval = crate::arkode_io::ARKodeSetAccumulatedErrorType(
                        arkode_mem,
                        ARK_ACCUMERROR_MAX,
                    );
                } else if argv[idx as usize] == *"ARK_ACCUMERROR_SUM" {
                    retval = crate::arkode_io::ARKodeSetAccumulatedErrorType(
                        arkode_mem,
                        ARK_ACCUMERROR_SUM,
                    );
                } else if argv[idx as usize] == *"ARK_ACCUMERROR_AVG" {
                    retval = crate::arkode_io::ARKodeSetAccumulatedErrorType(
                        arkode_mem,
                        ARK_ACCUMERROR_AVG,
                    );
                }
                if retval != ARK_SUCCESS {
                    arkProcessError(
                        Some(ark_mem),
                        retval,
                        line!() as i32,
                        "arkSetFromCommandLine",
                        file!(),
                        &format!(
                            "error setting key: {} {}",
                            argv[(idx - 1) as usize],
                            argv[idx as usize]
                        ),
                    );
                    return retval;
                }
                /* C's trailing `arg_used = SUNTRUE` is a dead store (see above) */
                break 'arg;
            }

            if argv[idx as usize][offset..] == *"write_parameters" {
                write_parameters = SUNTRUE;
                /* C's `arg_used = SUNTRUE` here is a dead store (see above) */
                break 'arg;
            }

            /* Call stepper-specific SetFromCommandLine routine (if supplied) to
            process this command-line argument */
            let step_setoptions = ark_mem.borrow().step_setoptions;
            if let Some(step_setoptions) = step_setoptions {
                let retval =
                    step_setoptions(ark_mem, &mut idx, argv, offset, &mut arg_used);
                if retval != ARK_SUCCESS {
                    return retval;
                }
                if arg_used {
                    break 'arg;
                }
            }

            /* warn for uninterpreted arkid.X arguments */
            arkProcessError(
                Some(ark_mem),
                ARK_WARNING,
                line!() as i32,
                "arkSetFromCommandLine",
                file!(),
                &format!("WARNING: key {} was not handled\n", argv[idx as usize]),
            );
        }
        idx += 1;
    }

    /* Call ARKodeWriteParameters (if requested) now that all
    command-line options have been set -- WARNING: this knows
    nothing about MPI, so it could be redundantly written by all
    processes if requested. */
    if write_parameters {
        let retval = crate::arkode_io::ARKodeWriteParameters(arkode_mem, &SUNFile::Stdout);
        if retval != ARK_SUCCESS {
            arkProcessError(
                Some(ark_mem),
                retval,
                line!() as i32,
                "arkSetFromCommandLine",
                file!(),
                "error writing parameters to stdout",
            );
            return retval;
        }
    }

    ARK_SUCCESS
}

/*===============================================================
  EOF
  ===============================================================*/
