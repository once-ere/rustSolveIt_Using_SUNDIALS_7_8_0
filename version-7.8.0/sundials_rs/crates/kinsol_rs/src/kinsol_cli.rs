//! Port of `src/kinsol/kinsol_cli.c` (command-line control over optional
//! inputs to KINSOL).
//!
//! The C key tables hold pointers to the public `KINSet*` setters, which
//! receive the raw `void* kinmem` forwarded by the `sunCheckAndSet*Args`
//! helpers. Here each table entry is a small adapter matching
//! `sundials_core::sundials_cli`'s setter fn types: it downcasts the
//! token (`Option<Box<dyn Any>>` holding a `KINMem` clone) back to the
//! handle and forwards to the real setter. Setters whose C parameter is
//! `sunbooleantype` (an `int` in C, fed directly from `atoi`) convert
//! with `arg != 0` — observably identical because C only truth-tests the
//! stored value.
//!
//! KINSOL declares no `sunKeyActionPair` table, so — unlike
//! `cvode_cli.rs` — there is no action-argument pass in the loop.

use std::any::Any;

use sundials_core::sundials_cli::*;
use sundials_core::sundials_types::*;

use crate::kinsol_impl::{KINMem, KINProcessError, KIN_ILL_INPUT, KIN_SUCCESS, KIN_WARNING};

/* -----------------------------------------------------------------
 * Adapter helpers: recover the KINMem handle from the CLI token
 * (C: the raw `void* kinmem` passed through sunCheckAndSet*Args).
 * A missing/mistyped token corresponds to C passing a garbage pointer
 * to the setter (UB) and maps to a deterministic panic.
 * ----------------------------------------------------------------- */

fn cliKINMem(mem: &mut Option<Box<dyn Any>>) -> KINMem {
    mem.as_mut()
        .and_then(|b| b.downcast_ref::<KINMem>())
        .cloned()
        .expect("kinmem token")
}

/* "int" setter adapters (table order below) */

fn cliKINSetOrthAA(mem: &mut Option<Box<dyn Any>>, arg: i32) -> i32 {
    let kin_mem = cliKINMem(mem);
    crate::kinsol_io::KINSetOrthAA(&kin_mem, arg)
}

fn cliKINSetReturnNewest(mem: &mut Option<Box<dyn Any>>, arg: i32) -> i32 {
    let kin_mem = cliKINMem(mem);
    /* C passes the int straight through as sunbooleantype */
    crate::kinsol_io::KINSetReturnNewest(&kin_mem, arg != 0)
}

fn cliKINSetNoInitSetup(mem: &mut Option<Box<dyn Any>>, arg: i32) -> i32 {
    let kin_mem = cliKINMem(mem);
    crate::kinsol_io::KINSetNoInitSetup(&kin_mem, arg != 0)
}

fn cliKINSetNoResMon(mem: &mut Option<Box<dyn Any>>, arg: i32) -> i32 {
    let kin_mem = cliKINMem(mem);
    crate::kinsol_io::KINSetNoResMon(&kin_mem, arg != 0)
}

fn cliKINSetEtaForm(mem: &mut Option<Box<dyn Any>>, arg: i32) -> i32 {
    let kin_mem = cliKINMem(mem);
    crate::kinsol_io::KINSetEtaForm(&kin_mem, arg)
}

fn cliKINSetNoMinEps(mem: &mut Option<Box<dyn Any>>, arg: i32) -> i32 {
    let kin_mem = cliKINMem(mem);
    crate::kinsol_io::KINSetNoMinEps(&kin_mem, arg != 0)
}

/* "long int" setter adapters (table order below) */

fn cliKINSetMAA(mem: &mut Option<Box<dyn Any>>, arg: i64) -> i32 {
    let kin_mem = cliKINMem(mem);
    crate::kinsol_io::KINSetMAA(&kin_mem, arg)
}

fn cliKINSetDelayAA(mem: &mut Option<Box<dyn Any>>, arg: i64) -> i32 {
    let kin_mem = cliKINMem(mem);
    crate::kinsol_io::KINSetDelayAA(&kin_mem, arg)
}

fn cliKINSetNumMaxIters(mem: &mut Option<Box<dyn Any>>, arg: i64) -> i32 {
    let kin_mem = cliKINMem(mem);
    crate::kinsol_io::KINSetNumMaxIters(&kin_mem, arg)
}

fn cliKINSetMaxSetupCalls(mem: &mut Option<Box<dyn Any>>, arg: i64) -> i32 {
    let kin_mem = cliKINMem(mem);
    crate::kinsol_io::KINSetMaxSetupCalls(&kin_mem, arg)
}

fn cliKINSetMaxSubSetupCalls(mem: &mut Option<Box<dyn Any>>, arg: i64) -> i32 {
    let kin_mem = cliKINMem(mem);
    crate::kinsol_io::KINSetMaxSubSetupCalls(&kin_mem, arg)
}

fn cliKINSetMaxBetaFails(mem: &mut Option<Box<dyn Any>>, arg: i64) -> i32 {
    let kin_mem = cliKINMem(mem);
    crate::kinsol_io::KINSetMaxBetaFails(&kin_mem, arg)
}

/* "sunrealtype" setter adapters (table order below) */

fn cliKINSetDamping(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let kin_mem = cliKINMem(mem);
    crate::kinsol_io::KINSetDamping(&kin_mem, arg)
}

fn cliKINSetDampingAA(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let kin_mem = cliKINMem(mem);
    crate::kinsol_io::KINSetDampingAA(&kin_mem, arg)
}

fn cliKINSetEtaConstValue(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let kin_mem = cliKINMem(mem);
    crate::kinsol_io::KINSetEtaConstValue(&kin_mem, arg)
}

fn cliKINSetResMonConstValue(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let kin_mem = cliKINMem(mem);
    crate::kinsol_io::KINSetResMonConstValue(&kin_mem, arg)
}

fn cliKINSetMaxNewtonStep(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let kin_mem = cliKINMem(mem);
    crate::kinsol_io::KINSetMaxNewtonStep(&kin_mem, arg)
}

fn cliKINSetRelErrFunc(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let kin_mem = cliKINMem(mem);
    crate::kinsol_io::KINSetRelErrFunc(&kin_mem, arg)
}

fn cliKINSetFuncNormTol(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let kin_mem = cliKINMem(mem);
    crate::kinsol_io::KINSetFuncNormTol(&kin_mem, arg)
}

fn cliKINSetScaledStepTol(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
    let kin_mem = cliKINMem(mem);
    crate::kinsol_io::KINSetScaledStepTol(&kin_mem, arg)
}

/* pair-of-sunrealtype setter adapters (table order below) */

fn cliKINSetEtaParams(
    mem: &mut Option<Box<dyn Any>>,
    arg1: sunrealtype,
    arg2: sunrealtype,
) -> i32 {
    let kin_mem = cliKINMem(mem);
    crate::kinsol_io::KINSetEtaParams(&kin_mem, arg1, arg2)
}

fn cliKINSetResMonParams(
    mem: &mut Option<Box<dyn Any>>,
    arg1: sunrealtype,
    arg2: sunrealtype,
) -> i32 {
    let kin_mem = cliKINMem(mem);
    crate::kinsol_io::KINSetResMonParams(&kin_mem, arg1, arg2)
}

/*---------------------------------------------------------------
  KINSetOptions:

  Sets KINSOL options using strings.
  ---------------------------------------------------------------*/

pub fn KINSetOptions(
    kinmem: &KINMem,
    kinid: Option<&str>,
    file_name: Option<&str>,
    argc: i32,
    argv: &[String],
) -> i32 {
    if let Some(file_name) = file_name {
        if !file_name.is_empty() {
            let retval = KIN_ILL_INPUT;
            KINProcessError(
                Some(kinmem),
                retval,
                line!() as i32,
                "KINSetOptions",
                file!(),
                "file-based options are not currently supported.",
            );
            return retval;
        }
    }

    if argc > 0 {
        /* C also checks argv != NULL; slices are never null */
        let retval = kinSetFromCommandLine(kinmem, kinid, argc, argv);
        if retval != KIN_SUCCESS {
            return retval;
        }
    }

    KIN_SUCCESS
}

/* -----------------------------------------------------------------
 * Function to control KINSOL options from the command line
 */

fn kinSetFromCommandLine(
    kinmem: &KINMem,
    kinid: Option<&str>,
    argc: i32,
    argv: &[String],
) -> i32 {
    /* NULL-mem check: handled by the type system */
    let kin_mem = kinmem;

    /* Set lists of command-line arguments, and the corresponding set routines */
    static int_pairs: [sunKeyIntPair; 6] = [
        sunKeyIntPair { key: "orth_aa", set: cliKINSetOrthAA },
        sunKeyIntPair { key: "return_newest", set: cliKINSetReturnNewest },
        sunKeyIntPair { key: "no_init_setup", set: cliKINSetNoInitSetup },
        sunKeyIntPair { key: "no_res_mon", set: cliKINSetNoResMon },
        sunKeyIntPair { key: "eta_form", set: cliKINSetEtaForm },
        sunKeyIntPair { key: "no_min_eps", set: cliKINSetNoMinEps },
    ];
    let num_int_keys: i32 = int_pairs.len() as i32;

    static long_pairs: [sunKeyLongPair; 6] = [
        sunKeyLongPair { key: "m_aa", set: cliKINSetMAA },
        sunKeyLongPair { key: "delay_aa", set: cliKINSetDelayAA },
        sunKeyLongPair { key: "num_max_iters", set: cliKINSetNumMaxIters },
        sunKeyLongPair { key: "max_setup_calls", set: cliKINSetMaxSetupCalls },
        sunKeyLongPair { key: "max_sub_setup_calls", set: cliKINSetMaxSubSetupCalls },
        sunKeyLongPair { key: "max_beta_fails", set: cliKINSetMaxBetaFails },
    ];
    let num_long_keys: i32 = long_pairs.len() as i32;

    static real_pairs: [sunKeyRealPair; 8] = [
        sunKeyRealPair { key: "damping", set: cliKINSetDamping },
        sunKeyRealPair { key: "damping_aa", set: cliKINSetDampingAA },
        sunKeyRealPair { key: "eta_const_value", set: cliKINSetEtaConstValue },
        sunKeyRealPair { key: "res_mon_const_value", set: cliKINSetResMonConstValue },
        sunKeyRealPair { key: "max_newton_step", set: cliKINSetMaxNewtonStep },
        sunKeyRealPair { key: "rel_err_func", set: cliKINSetRelErrFunc },
        sunKeyRealPair { key: "func_norm_tol", set: cliKINSetFuncNormTol },
        sunKeyRealPair { key: "scaled_step_tol", set: cliKINSetScaledStepTol },
    ];
    let num_real_keys: i32 = real_pairs.len() as i32;

    static tworeal_pairs: [sunKeyTwoRealPair; 2] = [
        sunKeyTwoRealPair { key: "eta_params", set: cliKINSetEtaParams },
        sunKeyTwoRealPair { key: "res_mon_params", set: cliKINSetResMonParams },
    ];
    let num_tworeal_keys: i32 = tworeal_pairs.len() as i32;

    /* Prefix for options to set */
    let default_id = "kinsol";
    let mut offset: usize = default_id.len() + 1;
    if let Some(kinid) = kinid {
        if !kinid.is_empty() {
            offset = kinid.len() + 1;
        }
    }
    let mut prefix = String::with_capacity(offset + 1);
    match kinid {
        Some(kinid) if !kinid.is_empty() => prefix.push_str(kinid),
        _ => prefix.push_str(default_id),
    }
    prefix.push('.');

    /* the CLI helpers receive C's `void* kinmem` as a boxed handle clone */
    let mut mem: Option<Box<dyn Any>> = Some(Box::new(kinmem.clone()));

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
            if retval != KIN_SUCCESS {
                KINProcessError(
                    Some(kin_mem),
                    retval,
                    line!() as i32,
                    "kinSetFromCommandLine",
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
            if retval != KIN_SUCCESS {
                KINProcessError(
                    Some(kin_mem),
                    retval,
                    line!() as i32,
                    "kinSetFromCommandLine",
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
            if retval != KIN_SUCCESS {
                KINProcessError(
                    Some(kin_mem),
                    retval,
                    line!() as i32,
                    "kinSetFromCommandLine",
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
            if retval != KIN_SUCCESS {
                KINProcessError(
                    Some(kin_mem),
                    retval,
                    line!() as i32,
                    "kinSetFromCommandLine",
                    file!(),
                    &format!("error setting key: {}", tworeal_pairs[j as usize].key),
                );
                return retval;
            }
            if arg_used {
                break 'arg;
            }

            /* warn for uninterpreted kinid.X arguments */
            KINProcessError(
                Some(kin_mem),
                KIN_WARNING,
                line!() as i32,
                "kinSetFromCommandLine",
                file!(),
                &format!("WARNING: key {} was not handled\n", argv[idx as usize]),
            );
        }
        idx += 1;
    }
    KIN_SUCCESS
}

/*===============================================================
  EOF
  ===============================================================*/
