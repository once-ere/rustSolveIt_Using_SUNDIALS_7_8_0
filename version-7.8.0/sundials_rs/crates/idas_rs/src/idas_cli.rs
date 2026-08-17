//! Port of `src/idas/idas_cli.c` (command-line control over optional
//! inputs to IDAS).
//!
//! The C key tables hold pointers to the public `IDASet*` setters, which
//! receive the raw `void* ida_mem` forwarded by the `sunCheckAndSet*Args`
//! helpers. Here each table entry is a small adapter matching
//! `sundials_core::sundials_cli`'s setter fn types: it downcasts the
//! token (`Option<Box<dyn Any>>` holding an `IDAMem` clone) back to the
//! handle and forwards to the real setter. Setters whose C parameter is
//! `sunbooleantype` (an `int` in C, fed directly from `atoi`) convert
//! with `arg != 0` — observably identical because C only truth-tests the
//! stored value. Setters whose C parameter is a plain `int` that the
//! callee later truth-tests (`IDASetQuadErrConB`) keep the `int`.
//!
//! Relative to IDA this file adds the sensitivity / quadrature keys plus
//! four whole key groups for the adjoint (`*_b`) setters: pair-of-int,
//! int+real, int+long and int+real+real.
//!
//! Cross-module note: the setters live in the sibling modules named
//! after their upstream C files — `idas_io.c`, `idas_ls.c`, `idaa_io.c`,
//! `idas.c` and `idaa.c`. If `idas.c` / `idaa.c` land as several
//! fragment modules, only the four `crate::idas::` / `crate::idaa::`
//! paths below (`IDASStolerances`, `IDAQuadSStolerances`,
//! `IDASensToggleOff` is in `idas.c`; `IDASStolerancesB` and
//! `IDAQuadSStolerancesB` are in `idaa.c`) need retargeting.

use std::any::Any;

use sundials_core::sundials_cli::*;
use sundials_core::sundials_types::*;

use crate::idas_impl::*;

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
    let ida_mem = cliIDAMem(mem);
    crate::idas_io::IDASetMaxNumStepsIC(&ida_mem, arg)
}

fn cliIDASetMaxNumJacsIC(mem: &mut Option<Box<dyn Any>>, arg: i32) -> i32 {
    let ida_mem = cliIDAMem(mem);
    crate::idas_io::IDASetMaxNumJacsIC(&ida_mem, arg)
}

fn cliIDASetMaxNumItersIC(mem: &mut Option<Box<dyn Any>>, arg: i32) -> i32 {
    let ida_mem = cliIDAMem(mem);
    crate::idas_io::IDASetMaxNumItersIC(&ida_mem, arg)
}

fn cliIDASetLineSearchOffIC(mem: &mut Option<Box<dyn Any>>, arg: i32) -> i32 {
    let ida_mem = cliIDAMem(mem);
    /* C passes the int straight through as sunbooleantype */
    crate::idas_io::IDASetLineSearchOffIC(&ida_mem, arg != 0)
}

fn cliIDASetMaxBacksIC(mem: &mut Option<Box<dyn Any>>, arg: i32) -> i32 {
    let ida_mem = cliIDAMem(mem);
    crate::idas_io::IDASetMaxBacksIC(&ida_mem, arg)
}

fn cliIDASetMaxOrd(mem: &mut Option<Box<dyn Any>>, arg: i32) -> i32 {
    let ida_mem = cliIDAMem(mem);
    crate::idas_io::IDASetMaxOrd(&ida_mem, arg)
}

fn cliIDASetMaxErrTestFails(mem: &mut Option<Box<dyn Any>>, arg: i32) -> i32 {
    let ida_mem = cliIDAMem(mem);
    crate::idas_io::IDASetMaxErrTestFails(&ida_mem, arg)
}

fn cliIDASetSuppressAlg(mem: &mut Option<Box<dyn Any>>, arg: i32) -> i32 {
    let ida_mem = cliIDAMem(mem);
    crate::idas_io::IDASetSuppressAlg(&ida_mem, arg != 0)
}

fn cliIDASetMaxConvFails(mem: &mut Option<Box<dyn Any>>, arg: i32) -> i32 {
    let ida_mem = cliIDAMem(mem);
    crate::idas_io::IDASetMaxConvFails(&ida_mem, arg)
}

fn cliIDASetMaxNonlinIters(mem: &mut Option<Box<dyn Any>>, arg: i32) -> i32 {
    let ida_mem = cliIDAMem(mem);
    crate::idas_io::IDASetMaxNonlinIters(&ida_mem, arg)
}

fn cliIDASetQuadErrCon(mem: &mut Option<Box<dyn Any>>, arg: i32) -> i32 {
    let ida_mem = cliIDAMem(mem);
    crate::idas_io::IDASetQuadErrCon(&ida_mem, arg != 0)
}

fn cliIDASetSensErrCon(mem: &mut Option<Box<dyn Any>>, arg: i32) -> i32 {
    let ida_mem = cliIDAMem(mem);
    crate::idas_io::IDASetSensErrCon(&ida_mem, arg != 0)
}

fn cliIDASetSensMaxNonlinIters(mem: &mut Option<Box<dyn Any>>, arg: i32) -> i32 {
    let ida_mem = cliIDAMem(mem);
    crate::idas_io::IDASetSensMaxNonlinIters(&ida_mem, arg)
}

fn cliIDASetQuadSensErrCon(mem: &mut Option<Box<dyn Any>>, arg: i32) -> i32 {
    let ida_mem = cliIDAMem(mem);
    crate::idas_io::IDASetQuadSensErrCon(&ida_mem, arg != 0)
}

fn cliIDASetLinearSolutionScaling(mem: &mut Option<Box<dyn Any>>, arg: i32) -> i32 {
    let ida_mem = cliIDAMem(mem);
    crate::idas_ls::IDASetLinearSolutionScaling(&ida_mem, arg != 0)
}

fn cliIDASetMaxNumConstraintFails(mem: &mut Option<Box<dyn Any>>, arg: i32) -> i32 {
    let ida_mem = cliIDAMem(mem);
    crate::idas_io::IDASetMaxNumConstraintFails(&ida_mem, arg)
}

/* "long int" setter adapters (table order below) */

fn cliIDASetMaxNumSteps(mem: &mut Option<Box<dyn Any>>, arg: i64) -> i32 {
    let ida_mem = cliIDAMem(mem);
    crate::idas_io::IDASetMaxNumSteps(&ida_mem, arg)
}

/* "sunrealtype" setter adapters (table order below) */

fn cliIDASetNonlinConvCoefIC(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let ida_mem = cliIDAMem(mem);
    crate::idas_io::IDASetNonlinConvCoefIC(&ida_mem, arg)
}

fn cliIDASetStepToleranceIC(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let ida_mem = cliIDAMem(mem);
    crate::idas_io::IDASetStepToleranceIC(&ida_mem, arg)
}

fn cliIDASetDeltaCjLSetup(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let ida_mem = cliIDAMem(mem);
    crate::idas_io::IDASetDeltaCjLSetup(&ida_mem, arg)
}

fn cliIDASetInitStep(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let ida_mem = cliIDAMem(mem);
    crate::idas_io::IDASetInitStep(&ida_mem, arg)
}

fn cliIDASetMaxStep(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let ida_mem = cliIDAMem(mem);
    crate::idas_io::IDASetMaxStep(&ida_mem, arg)
}

fn cliIDASetMinStep(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let ida_mem = cliIDAMem(mem);
    crate::idas_io::IDASetMinStep(&ida_mem, arg)
}

fn cliIDASetStopTime(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let ida_mem = cliIDAMem(mem);
    crate::idas_io::IDASetStopTime(&ida_mem, arg)
}

fn cliIDASetEtaMin(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let ida_mem = cliIDAMem(mem);
    crate::idas_io::IDASetEtaMin(&ida_mem, arg)
}

fn cliIDASetEtaMax(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let ida_mem = cliIDAMem(mem);
    crate::idas_io::IDASetEtaMax(&ida_mem, arg)
}

fn cliIDASetEtaLow(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let ida_mem = cliIDAMem(mem);
    crate::idas_io::IDASetEtaLow(&ida_mem, arg)
}

fn cliIDASetEtaMinErrFail(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let ida_mem = cliIDAMem(mem);
    crate::idas_io::IDASetEtaMinErrFail(&ida_mem, arg)
}

fn cliIDASetEtaConvFail(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let ida_mem = cliIDAMem(mem);
    crate::idas_io::IDASetEtaConvFail(&ida_mem, arg)
}

fn cliIDASetNonlinConvCoef(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let ida_mem = cliIDAMem(mem);
    crate::idas_io::IDASetNonlinConvCoef(&ida_mem, arg)
}

fn cliIDASetEpsLin(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let ida_mem = cliIDAMem(mem);
    crate::idas_ls::IDASetEpsLin(&ida_mem, arg)
}

fn cliIDASetLSNormFactor(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let ida_mem = cliIDAMem(mem);
    crate::idas_ls::IDASetLSNormFactor(&ida_mem, arg)
}

fn cliIDASetIncrementFactor(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let ida_mem = cliIDAMem(mem);
    crate::idas_ls::IDASetIncrementFactor(&ida_mem, arg)
}

/* pair-of-sunrealtype setter adapters (table order below) */

fn cliIDASetEtaFixedStepBounds(
    mem: &mut Option<Box<dyn Any>>,
    arg1: sunrealtype,
    arg2: sunrealtype,
) -> i32 {
    let ida_mem = cliIDAMem(mem);
    crate::idas_io::IDASetEtaFixedStepBounds(&ida_mem, arg1, arg2)
}

fn cliIDASStolerances(
    mem: &mut Option<Box<dyn Any>>,
    arg1: sunrealtype,
    arg2: sunrealtype,
) -> i32 {
    let ida_mem = cliIDAMem(mem);
    crate::idas::IDASStolerances(&ida_mem, arg1, arg2)
}

fn cliIDAQuadSStolerances(
    mem: &mut Option<Box<dyn Any>>,
    arg1: sunrealtype,
    arg2: sunrealtype,
) -> i32 {
    let ida_mem = cliIDAMem(mem);
    crate::idas::IDAQuadSStolerances(&ida_mem, arg1, arg2)
}

/* pair-of-int setter adapters (table order below) */

fn cliIDASetMaxOrdB(mem: &mut Option<Box<dyn Any>>, arg1: i32, arg2: i32) -> i32 {
    let ida_mem = cliIDAMem(mem);
    crate::idaa_io::IDASetMaxOrdB(&ida_mem, arg1, arg2)
}

fn cliIDASetSuppressAlgB(mem: &mut Option<Box<dyn Any>>, arg1: i32, arg2: i32) -> i32 {
    let ida_mem = cliIDAMem(mem);
    crate::idaa_io::IDASetSuppressAlgB(&ida_mem, arg1, arg2 != 0)
}

fn cliIDASetQuadErrConB(mem: &mut Option<Box<dyn Any>>, arg1: i32, arg2: i32) -> i32 {
    let ida_mem = cliIDAMem(mem);
    /* C signature takes a plain `int` here (unlike IDASetSuppressAlgB) */
    crate::idaa_io::IDASetQuadErrConB(&ida_mem, arg1, arg2 != 0)
}

fn cliIDASetLinearSolutionScalingB(mem: &mut Option<Box<dyn Any>>, arg1: i32, arg2: i32) -> i32 {
    let ida_mem = cliIDAMem(mem);
    crate::idas_ls::IDASetLinearSolutionScalingB(&ida_mem, arg1, arg2 != 0)
}

/* action setter adapters (table order below) */

fn cliIDAClearStopTime(mem: &mut Option<Box<dyn Any>>) -> i32 {
    let ida_mem = cliIDAMem(mem);
    crate::idas_io::IDAClearStopTime(&ida_mem)
}

fn cliIDASetNoInactiveRootWarn(mem: &mut Option<Box<dyn Any>>) -> i32 {
    let ida_mem = cliIDAMem(mem);
    crate::idas_io::IDASetNoInactiveRootWarn(&ida_mem)
}

fn cliIDASensToggleOff(mem: &mut Option<Box<dyn Any>>) -> i32 {
    let ida_mem = cliIDAMem(mem);
    crate::idas::IDASensToggleOff(&ida_mem)
}

fn cliIDAAdjSetNoSensi(mem: &mut Option<Box<dyn Any>>) -> i32 {
    let ida_mem = cliIDAMem(mem);
    crate::idaa_io::IDAAdjSetNoSensi(&ida_mem)
}

/* int+real setter adapters (table order below) */

fn cliIDASetSensDQMethod(mem: &mut Option<Box<dyn Any>>, arg1: i32, arg2: sunrealtype) -> i32 {
    let ida_mem = cliIDAMem(mem);
    crate::idas_io::IDASetSensDQMethod(&ida_mem, arg1, arg2)
}

fn cliIDASetInitStepB(mem: &mut Option<Box<dyn Any>>, arg1: i32, arg2: sunrealtype) -> i32 {
    let ida_mem = cliIDAMem(mem);
    crate::idaa_io::IDASetInitStepB(&ida_mem, arg1, arg2)
}

fn cliIDASetMaxStepB(mem: &mut Option<Box<dyn Any>>, arg1: i32, arg2: sunrealtype) -> i32 {
    let ida_mem = cliIDAMem(mem);
    crate::idaa_io::IDASetMaxStepB(&ida_mem, arg1, arg2)
}

fn cliIDASetEpsLinB(mem: &mut Option<Box<dyn Any>>, arg1: i32, arg2: sunrealtype) -> i32 {
    let ida_mem = cliIDAMem(mem);
    crate::idas_ls::IDASetEpsLinB(&ida_mem, arg1, arg2)
}

fn cliIDASetLSNormFactorB(mem: &mut Option<Box<dyn Any>>, arg1: i32, arg2: sunrealtype) -> i32 {
    let ida_mem = cliIDAMem(mem);
    crate::idas_ls::IDASetLSNormFactorB(&ida_mem, arg1, arg2)
}

fn cliIDASetIncrementFactorB(mem: &mut Option<Box<dyn Any>>, arg1: i32, arg2: sunrealtype) -> i32 {
    let ida_mem = cliIDAMem(mem);
    crate::idas_ls::IDASetIncrementFactorB(&ida_mem, arg1, arg2)
}

/* int+real+real setter adapters (table order below) */

fn cliIDASStolerancesB(
    mem: &mut Option<Box<dyn Any>>,
    arg1: i32,
    arg2: sunrealtype,
    arg3: sunrealtype,
) -> i32 {
    let ida_mem = cliIDAMem(mem);
    crate::idaa::IDASStolerancesB(&ida_mem, arg1, arg2, arg3)
}

fn cliIDAQuadSStolerancesB(
    mem: &mut Option<Box<dyn Any>>,
    arg1: i32,
    arg2: sunrealtype,
    arg3: sunrealtype,
) -> i32 {
    let ida_mem = cliIDAMem(mem);
    crate::idaa::IDAQuadSStolerancesB(&ida_mem, arg1, arg2, arg3)
}

/* int+long setter adapters (table order below) */

fn cliIDASetMaxNumStepsB(mem: &mut Option<Box<dyn Any>>, arg1: i32, arg2: i64) -> i32 {
    let ida_mem = cliIDAMem(mem);
    crate::idaa_io::IDASetMaxNumStepsB(&ida_mem, arg1, arg2)
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
    static int_pairs: [sunKeyIntPair; 16] = [
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
        sunKeyIntPair { key: "quad_err_con", set: cliIDASetQuadErrCon },
        sunKeyIntPair { key: "sens_err_con", set: cliIDASetSensErrCon },
        sunKeyIntPair { key: "sens_max_nonlin_iters", set: cliIDASetSensMaxNonlinIters },
        sunKeyIntPair { key: "quad_sens_err_con", set: cliIDASetQuadSensErrCon },
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

    static tworeal_pairs: [sunKeyTwoRealPair; 3] = [
        sunKeyTwoRealPair { key: "eta_fixed_step_bounds", set: cliIDASetEtaFixedStepBounds },
        sunKeyTwoRealPair { key: "scalar_tolerances", set: cliIDASStolerances },
        sunKeyTwoRealPair { key: "quad_scalar_tolerances", set: cliIDAQuadSStolerances },
    ];
    let num_tworeal_keys: i32 = tworeal_pairs.len() as i32;

    static twoint_pairs: [sunKeyTwoIntPair; 4] = [
        sunKeyTwoIntPair { key: "max_order_b", set: cliIDASetMaxOrdB },
        sunKeyTwoIntPair { key: "suppress_alg_b", set: cliIDASetSuppressAlgB },
        sunKeyTwoIntPair { key: "quad_err_con_b", set: cliIDASetQuadErrConB },
        sunKeyTwoIntPair {
            key: "linear_solution_scaling_b",
            set: cliIDASetLinearSolutionScalingB,
        },
    ];
    let num_twoint_keys: i32 = twoint_pairs.len() as i32;

    static action_pairs: [sunKeyActionPair; 4] = [
        sunKeyActionPair { key: "clear_stop_time", set: cliIDAClearStopTime },
        sunKeyActionPair { key: "no_inactive_root_warn", set: cliIDASetNoInactiveRootWarn },
        sunKeyActionPair { key: "sens_toggle_off", set: cliIDASensToggleOff },
        sunKeyActionPair { key: "adj_no_sensi", set: cliIDAAdjSetNoSensi },
    ];
    let num_action_keys: i32 = action_pairs.len() as i32;

    static int_real_pairs: [sunKeyIntRealPair; 6] = [
        sunKeyIntRealPair { key: "sens_dq_method", set: cliIDASetSensDQMethod },
        sunKeyIntRealPair { key: "init_step_b", set: cliIDASetInitStepB },
        sunKeyIntRealPair { key: "max_step_b", set: cliIDASetMaxStepB },
        sunKeyIntRealPair { key: "eps_lin_b", set: cliIDASetEpsLinB },
        sunKeyIntRealPair { key: "ls_norm_factor_b", set: cliIDASetLSNormFactorB },
        sunKeyIntRealPair { key: "increment_factor_b", set: cliIDASetIncrementFactorB },
    ];
    let num_int_real_keys: i32 = int_real_pairs.len() as i32;

    static int_real_real_pairs: [sunKeyIntRealRealPair; 2] = [
        sunKeyIntRealRealPair { key: "scalar_tolerances_b", set: cliIDASStolerancesB },
        sunKeyIntRealRealPair {
            key: "quad_scalar_tolerances_b",
            set: cliIDAQuadSStolerancesB,
        },
    ];
    let num_int_real_real_keys: i32 = int_real_real_pairs.len() as i32;

    static int_long_pairs: [sunKeyIntLongPair; 1] =
        [sunKeyIntLongPair { key: "max_num_steps_b", set: cliIDASetMaxNumStepsB }];
    let num_int_long_keys: i32 = int_long_pairs.len() as i32;

    /* Prefix for options to set */
    let default_id = "idas";
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

            /* check all pair-of-int command-line options */
            let retval = sunCheckAndSetTwoIntArgs(&mut mem, &mut idx, argv, offset, &twoint_pairs,
                                                  num_twoint_keys, &mut arg_used, &mut j);
            if retval != IDA_SUCCESS {
                IDAProcessError(
                    Some(IDA_mem),
                    retval,
                    line!() as i32,
                    "idaSetFromCommandLine",
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

            /* check all int+real command-line options */
            let retval = sunCheckAndSetIntRealArgs(&mut mem, &mut idx, argv, offset,
                                                   &int_real_pairs, num_int_real_keys,
                                                   &mut arg_used, &mut j);
            if retval != IDA_SUCCESS {
                IDAProcessError(
                    Some(IDA_mem),
                    retval,
                    line!() as i32,
                    "idaSetFromCommandLine",
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
            if retval != IDA_SUCCESS {
                IDAProcessError(
                    Some(IDA_mem),
                    retval,
                    line!() as i32,
                    "idaSetFromCommandLine",
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
            if retval != IDA_SUCCESS {
                IDAProcessError(
                    Some(IDA_mem),
                    retval,
                    line!() as i32,
                    "idaSetFromCommandLine",
                    file!(),
                    &format!("error setting key: {}", int_real_real_pairs[j as usize].key),
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
