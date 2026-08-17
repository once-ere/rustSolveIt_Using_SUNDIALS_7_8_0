//! Port of `src/ida/ida_cli.c` (command-line control over optional
//! inputs to IDA).
//!
//! The C key tables hold pointers to the public `IDASet*` setters, which
//! receive the raw `void* ida_mem` forwarded by the `sunCheckAndSet*Args`
//! helpers. Here each table entry is a small adapter matching
//! `sundials_core::sundials_cli`'s setter fn types: it downcasts the
//! token (`Option<Box<dyn Any>>` holding an `IDAMem` clone) back to the
//! handle and forwards to the real setter. Setters whose C parameter is
//! `sunbooleantype` (an `int` in C, fed directly from `atoi`) convert
//! with `arg != 0` — observably identical because C only truth-tests the
//! stored value.
//!
//! The tables keep the upstream order byte-for-byte: `sunCheckAndSet*Args`
//! scans linearly and stops at the first match, and the `j` index it
//! reports on failure indexes the same table.

use std::any::Any;

use sundials_core::sundials_cli::*;
use sundials_core::sundials_types::*;

use crate::ida_impl::*;

/* -----------------------------------------------------------------
 * Adapter helpers: recover the IDAMem handle from the CLI token
 * (C: the raw `void* ida_mem` passed through sunCheckAndSet*Args).
 * A missing/mistyped token corresponds to C passing a garbage pointer
 * to the setter (UB) and maps to a deterministic panic.
 * ----------------------------------------------------------------- */

fn cliIDAMem(mem: &mut Option<Box<dyn Any>>) -> IDAMem {
    mem.as_mut()
        .and_then(|b| b.downcast_ref::<IDAMem>())
        .cloned()
        .expect("ida_mem token")
}

/* "int" setter adapters (table order below) */

fn cliIDASetMaxNumStepsIC(mem: &mut Option<Box<dyn Any>>, arg: i32) -> i32 {
    let IDA_mem = cliIDAMem(mem);
    crate::ida_io::IDASetMaxNumStepsIC(&IDA_mem, arg)
}

fn cliIDASetMaxNumJacsIC(mem: &mut Option<Box<dyn Any>>, arg: i32) -> i32 {
    let IDA_mem = cliIDAMem(mem);
    crate::ida_io::IDASetMaxNumJacsIC(&IDA_mem, arg)
}

fn cliIDASetMaxNumItersIC(mem: &mut Option<Box<dyn Any>>, arg: i32) -> i32 {
    let IDA_mem = cliIDAMem(mem);
    crate::ida_io::IDASetMaxNumItersIC(&IDA_mem, arg)
}

fn cliIDASetLineSearchOffIC(mem: &mut Option<Box<dyn Any>>, arg: i32) -> i32 {
    let IDA_mem = cliIDAMem(mem);
    /* C passes the int straight through as sunbooleantype */
    crate::ida_io::IDASetLineSearchOffIC(&IDA_mem, arg != 0)
}

fn cliIDASetMaxBacksIC(mem: &mut Option<Box<dyn Any>>, arg: i32) -> i32 {
    let IDA_mem = cliIDAMem(mem);
    crate::ida_io::IDASetMaxBacksIC(&IDA_mem, arg)
}

fn cliIDASetMaxOrd(mem: &mut Option<Box<dyn Any>>, arg: i32) -> i32 {
    let IDA_mem = cliIDAMem(mem);
    crate::ida_io::IDASetMaxOrd(&IDA_mem, arg)
}

fn cliIDASetMaxErrTestFails(mem: &mut Option<Box<dyn Any>>, arg: i32) -> i32 {
    let IDA_mem = cliIDAMem(mem);
    crate::ida_io::IDASetMaxErrTestFails(&IDA_mem, arg)
}

fn cliIDASetSuppressAlg(mem: &mut Option<Box<dyn Any>>, arg: i32) -> i32 {
    let IDA_mem = cliIDAMem(mem);
    crate::ida_io::IDASetSuppressAlg(&IDA_mem, arg != 0)
}

fn cliIDASetMaxConvFails(mem: &mut Option<Box<dyn Any>>, arg: i32) -> i32 {
    let IDA_mem = cliIDAMem(mem);
    crate::ida_io::IDASetMaxConvFails(&IDA_mem, arg)
}

fn cliIDASetMaxNonlinIters(mem: &mut Option<Box<dyn Any>>, arg: i32) -> i32 {
    let IDA_mem = cliIDAMem(mem);
    crate::ida_io::IDASetMaxNonlinIters(&IDA_mem, arg)
}

fn cliIDASetLinearSolutionScaling(mem: &mut Option<Box<dyn Any>>, arg: i32) -> i32 {
    let IDA_mem = cliIDAMem(mem);
    crate::ida_ls::IDASetLinearSolutionScaling(&IDA_mem, arg != 0)
}

fn cliIDASetMaxNumConstraintFails(mem: &mut Option<Box<dyn Any>>, arg: i32) -> i32 {
    let IDA_mem = cliIDAMem(mem);
    crate::ida_io::IDASetMaxNumConstraintFails(&IDA_mem, arg)
}

/* "long int" setter adapters (table order below) */

fn cliIDASetMaxNumSteps(mem: &mut Option<Box<dyn Any>>, arg: i64) -> i32 {
    let IDA_mem = cliIDAMem(mem);
    crate::ida_io::IDASetMaxNumSteps(&IDA_mem, arg)
}

/* "sunrealtype" setter adapters (table order below) */

fn cliIDASetNonlinConvCoefIC(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let IDA_mem = cliIDAMem(mem);
    crate::ida_io::IDASetNonlinConvCoefIC(&IDA_mem, arg)
}

fn cliIDASetStepToleranceIC(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let IDA_mem = cliIDAMem(mem);
    crate::ida_io::IDASetStepToleranceIC(&IDA_mem, arg)
}

fn cliIDASetDeltaCjLSetup(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let IDA_mem = cliIDAMem(mem);
    crate::ida_io::IDASetDeltaCjLSetup(&IDA_mem, arg)
}

fn cliIDASetInitStep(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let IDA_mem = cliIDAMem(mem);
    crate::ida_io::IDASetInitStep(&IDA_mem, arg)
}

fn cliIDASetMaxStep(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let IDA_mem = cliIDAMem(mem);
    crate::ida_io::IDASetMaxStep(&IDA_mem, arg)
}

fn cliIDASetMinStep(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let IDA_mem = cliIDAMem(mem);
    crate::ida_io::IDASetMinStep(&IDA_mem, arg)
}

fn cliIDASetStopTime(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let IDA_mem = cliIDAMem(mem);
    crate::ida_io::IDASetStopTime(&IDA_mem, arg)
}

fn cliIDASetEtaMin(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let IDA_mem = cliIDAMem(mem);
    crate::ida_io::IDASetEtaMin(&IDA_mem, arg)
}

fn cliIDASetEtaMax(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let IDA_mem = cliIDAMem(mem);
    crate::ida_io::IDASetEtaMax(&IDA_mem, arg)
}

fn cliIDASetEtaLow(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let IDA_mem = cliIDAMem(mem);
    crate::ida_io::IDASetEtaLow(&IDA_mem, arg)
}

fn cliIDASetEtaMinErrFail(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let IDA_mem = cliIDAMem(mem);
    crate::ida_io::IDASetEtaMinErrFail(&IDA_mem, arg)
}

fn cliIDASetEtaConvFail(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let IDA_mem = cliIDAMem(mem);
    crate::ida_io::IDASetEtaConvFail(&IDA_mem, arg)
}

fn cliIDASetNonlinConvCoef(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let IDA_mem = cliIDAMem(mem);
    crate::ida_io::IDASetNonlinConvCoef(&IDA_mem, arg)
}

fn cliIDASetEpsLin(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let IDA_mem = cliIDAMem(mem);
    crate::ida_ls::IDASetEpsLin(&IDA_mem, arg)
}

fn cliIDASetLSNormFactor(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let IDA_mem = cliIDAMem(mem);
    crate::ida_ls::IDASetLSNormFactor(&IDA_mem, arg)
}

fn cliIDASetIncrementFactor(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let IDA_mem = cliIDAMem(mem);
    crate::ida_ls::IDASetIncrementFactor(&IDA_mem, arg)
}

/* pair-of-sunrealtype setter adapters (table order below) */

fn cliIDASetEtaFixedStepBounds(
    mem: &mut Option<Box<dyn Any>>,
    arg1: sunrealtype,
    arg2: sunrealtype,
) -> i32 {
    let IDA_mem = cliIDAMem(mem);
    crate::ida_io::IDASetEtaFixedStepBounds(&IDA_mem, arg1, arg2)
}

fn cliIDASStolerances(
    mem: &mut Option<Box<dyn Any>>,
    arg1: sunrealtype,
    arg2: sunrealtype,
) -> i32 {
    let IDA_mem = cliIDAMem(mem);
    crate::ida::IDASStolerances(&IDA_mem, arg1, arg2)
}

/* action setter adapters (table order below) */

fn cliIDAClearStopTime(mem: &mut Option<Box<dyn Any>>) -> i32 {
    let IDA_mem = cliIDAMem(mem);
    crate::ida_io::IDAClearStopTime(&IDA_mem)
}

fn cliIDASetNoInactiveRootWarn(mem: &mut Option<Box<dyn Any>>) -> i32 {
    let IDA_mem = cliIDAMem(mem);
    crate::ida_io::IDASetNoInactiveRootWarn(&IDA_mem)
}

/*---------------------------------------------------------------
  IDASetOptions:

  Sets IDA options using strings.
  ---------------------------------------------------------------*/

pub fn IDASetOptions(
    ida_mem: &IDAMem,
    idaid: Option<&str>,
    file_name: Option<&str>,
    argc: i32,
    argv: &[String],
) -> i32 {
    if let Some(file_name) = file_name {
        if !file_name.is_empty() {
            let retval = IDA_ILL_INPUT;
            IDAProcessError(
                Some(ida_mem),
                retval,
                line!() as i32,
                "IDASetOptions",
                file!(),
                "file-based options are not currently supported.",
            );
            return retval;
        }
    }

    if argc > 0 {
        /* C also checks argv != NULL; slices are never null */
        let retval = idaSetFromCommandLine(ida_mem, idaid, argc, argv);
        if retval != IDA_SUCCESS {
            return retval;
        }
    }

    IDA_SUCCESS
}

/* -----------------------------------------------------------------
 * Function to control IDA options from the command line
 */

fn idaSetFromCommandLine(
    ida_mem: &IDAMem,
    idaid: Option<&str>,
    argc: i32,
    argv: &[String],
) -> i32 {
    /* NULL-mem check: handled by type system */
    let IDA_mem = ida_mem;

    /* Set lists of command-line arguments, and the corresponding set routines */
    static int_pairs: [sunKeyIntPair; 12] = [
        sunKeyIntPair { key: "max_num_steps_ic", set: cliIDASetMaxNumStepsIC },
        sunKeyIntPair { key: "max_num_jacs_ic", set: cliIDASetMaxNumJacsIC },
        sunKeyIntPair { key: "max_num_iters_ic", set: cliIDASetMaxNumItersIC },
        sunKeyIntPair { key: "line_search_off_ic", set: cliIDASetLineSearchOffIC },
        sunKeyIntPair { key: "max_backs_ic", set: cliIDASetMaxBacksIC },
        sunKeyIntPair { key: "max_order", set: cliIDASetMaxOrd },
        sunKeyIntPair { key: "max_err_test_fails", set: cliIDASetMaxErrTestFails },
        sunKeyIntPair { key: "suppress_alg", set: cliIDASetSuppressAlg },
        sunKeyIntPair { key: "max_conv_fails", set: cliIDASetMaxConvFails },
        sunKeyIntPair { key: "max_nonlin_iters", set: cliIDASetMaxNonlinIters },
        sunKeyIntPair { key: "linear_solution_scaling", set: cliIDASetLinearSolutionScaling },
        sunKeyIntPair {
            key: "max_num_constraint_fails",
            set: cliIDASetMaxNumConstraintFails,
        },
    ];
    let num_int_keys: i32 = int_pairs.len() as i32;

    static long_pairs: [sunKeyLongPair; 1] =
        [sunKeyLongPair { key: "max_num_steps", set: cliIDASetMaxNumSteps }];
    let num_long_keys: i32 = long_pairs.len() as i32;

    static real_pairs: [sunKeyRealPair; 16] = [
        sunKeyRealPair { key: "nonlin_conv_coef_ic", set: cliIDASetNonlinConvCoefIC },
        sunKeyRealPair { key: "step_tolerance_ic", set: cliIDASetStepToleranceIC },
        sunKeyRealPair { key: "delta_cj_lsetup", set: cliIDASetDeltaCjLSetup },
        sunKeyRealPair { key: "init_step", set: cliIDASetInitStep },
        sunKeyRealPair { key: "max_step", set: cliIDASetMaxStep },
        sunKeyRealPair { key: "min_step", set: cliIDASetMinStep },
        sunKeyRealPair { key: "stop_time", set: cliIDASetStopTime },
        sunKeyRealPair { key: "eta_min", set: cliIDASetEtaMin },
        sunKeyRealPair { key: "eta_max", set: cliIDASetEtaMax },
        sunKeyRealPair { key: "eta_low", set: cliIDASetEtaLow },
        sunKeyRealPair { key: "eta_min_err_fail", set: cliIDASetEtaMinErrFail },
        sunKeyRealPair { key: "eta_conv_fail", set: cliIDASetEtaConvFail },
        sunKeyRealPair { key: "nonlin_conv_coef", set: cliIDASetNonlinConvCoef },
        sunKeyRealPair { key: "eps_lin", set: cliIDASetEpsLin },
        sunKeyRealPair { key: "ls_norm_factor", set: cliIDASetLSNormFactor },
        sunKeyRealPair { key: "increment_factor", set: cliIDASetIncrementFactor },
    ];
    let num_real_keys: i32 = real_pairs.len() as i32;

    static tworeal_pairs: [sunKeyTwoRealPair; 2] = [
        sunKeyTwoRealPair { key: "eta_fixed_step_bounds", set: cliIDASetEtaFixedStepBounds },
        sunKeyTwoRealPair { key: "scalar_tolerances", set: cliIDASStolerances },
    ];
    let num_tworeal_keys: i32 = tworeal_pairs.len() as i32;

    static action_pairs: [sunKeyActionPair; 2] = [
        sunKeyActionPair { key: "clear_stop_time", set: cliIDAClearStopTime },
        sunKeyActionPair { key: "no_inactive_root_warn", set: cliIDASetNoInactiveRootWarn },
    ];
    let num_action_keys: i32 = action_pairs.len() as i32;

    /* Prefix for options to set */
    let default_id = "ida";
    let mut offset: usize = default_id.len() + 1;
    if let Some(idaid) = idaid {
        if !idaid.is_empty() {
            offset = idaid.len() + 1;
        }
    }
    let mut prefix = String::with_capacity(offset + 1);
    match idaid {
        Some(idaid) if !idaid.is_empty() => prefix.push_str(idaid),
        _ => prefix.push_str(default_id),
    }
    prefix.push('.');

    /* the CLI helpers receive C's `void* ida_mem` as a boxed handle clone */
    let mut mem: Option<Box<dyn Any>> = Some(Box::new(ida_mem.clone()));

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
            if retval != IDA_SUCCESS {
                IDAProcessError(
                    Some(IDA_mem),
                    retval,
                    line!() as i32,
                    "idaSetFromCommandLine",
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
            if retval != IDA_SUCCESS {
                IDAProcessError(
                    Some(IDA_mem),
                    retval,
                    line!() as i32,
                    "idaSetFromCommandLine",
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
            if retval != IDA_SUCCESS {
                IDAProcessError(
                    Some(IDA_mem),
                    retval,
                    line!() as i32,
                    "idaSetFromCommandLine",
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
            if retval != IDA_SUCCESS {
                IDAProcessError(
                    Some(IDA_mem),
                    retval,
                    line!() as i32,
                    "idaSetFromCommandLine",
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
            if retval != IDA_SUCCESS {
                IDAProcessError(
                    Some(IDA_mem),
                    retval,
                    line!() as i32,
                    "idaSetFromCommandLine",
                    file!(),
                    &format!("error setting key: {}", action_pairs[j as usize].key),
                );
                return retval;
            }
            if arg_used {
                break 'arg;
            }

            /* warn for uninterpreted idaid.X arguments */
            IDAProcessError(
                Some(IDA_mem),
                IDA_WARNING,
                line!() as i32,
                "idaSetFromCommandLine",
                file!(),
                &format!("WARNING: key {} was not handled\n", argv[idx as usize]),
            );
        }
        idx += 1;
    }
    IDA_SUCCESS
}

/*===============================================================
  EOF
  ===============================================================*/
