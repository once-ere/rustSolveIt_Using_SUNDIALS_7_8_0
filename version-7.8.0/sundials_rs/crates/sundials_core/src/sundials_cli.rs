//! Port of `src/sundials/sundials_cli.c` + `src/sundials/sundials_cli.h`
//! (command-line control over optional solver inputs).
//!
//! Mapping notes: C `void* mem` setter targets become
//! `&mut Option<Box<dyn Any>>` (the take/restore token pattern); each C
//! function-pointer typedef is a Rust `type` alias over plain `fn`
//! pointers; `argv` is `&[String]` and `argidx` stays `&mut i32` (C
//! `int*`), cast with `as usize` only at indexing sites. Key matching is
//! C `strcmp(argv[*argidx] + offset, key) == 0`, i.e. a literal
//! comparison of the argument with the bare `<id>.` prefix stripped —
//! no leading dashes are involved anywhere.

use std::any::Any;

use crate::sundials_errors::SUN_SUCCESS;
use crate::sundials_math::SUNStrToReal;
use crate::sundials_types::{sunbooleantype, sunrealtype, SUNFALSE, SUNTRUE};

/*===============================================================
  Command-line input utility routines
  ===============================================================*/

/* utilities for integer "set" routines */
pub type sunIntSetFn = fn(mem: &mut Option<Box<dyn Any>>, arg: i32) -> i32;

#[derive(Clone, Copy)]
pub struct sunKeyIntPair {
    pub key: &'static str,
    pub set: sunIntSetFn,
}

pub fn sunCheckAndSetIntArgs(
    mem: &mut Option<Box<dyn Any>>,
    argidx: &mut i32,
    argv: &[String],
    offset: usize,
    testpairs: &[sunKeyIntPair],
    numpairs: i32,
    arg_used: &mut sunbooleantype,
    failedarg: &mut i32,
) -> i32 {
    for j in 0..numpairs {
        *arg_used = SUNFALSE;
        if argv[*argidx as usize][offset..] == *testpairs[j as usize].key {
            *argidx += 1;
            let iarg: i32 = crate::sundials_utils::atoi(&argv[*argidx as usize]);
            let retval = (testpairs[j as usize].set)(mem, iarg);
            if retval != SUN_SUCCESS {
                *failedarg = j;
                return retval;
            }
            *arg_used = SUNTRUE;
            return SUN_SUCCESS;
        }
    }
    SUN_SUCCESS
}

/* utilities for pair-of-integer "set" routines */
pub type sunTwoIntSetFn = fn(mem: &mut Option<Box<dyn Any>>, arg1: i32, arg2: i32) -> i32;

#[derive(Clone, Copy)]
pub struct sunKeyTwoIntPair {
    pub key: &'static str,
    pub set: sunTwoIntSetFn,
}

pub fn sunCheckAndSetTwoIntArgs(
    mem: &mut Option<Box<dyn Any>>,
    argidx: &mut i32,
    argv: &[String],
    offset: usize,
    testpairs: &[sunKeyTwoIntPair],
    numpairs: i32,
    arg_used: &mut sunbooleantype,
    failedarg: &mut i32,
) -> i32 {
    for j in 0..numpairs {
        *arg_used = SUNFALSE;
        if argv[*argidx as usize][offset..] == *testpairs[j as usize].key {
            *argidx += 1;
            let iarg1: i32 = crate::sundials_utils::atoi(&argv[*argidx as usize]);
            *argidx += 1;
            let iarg2: i32 = crate::sundials_utils::atoi(&argv[*argidx as usize]);
            let retval = (testpairs[j as usize].set)(mem, iarg1, iarg2);
            if retval != SUN_SUCCESS {
                *failedarg = j;
                return retval;
            }
            *arg_used = SUNTRUE;
            return SUN_SUCCESS;
        }
    }
    SUN_SUCCESS
}

/* utilities for long int "set" routines */
pub type sunLongSetFn = fn(mem: &mut Option<Box<dyn Any>>, arg: i64) -> i32;

#[derive(Clone, Copy)]
pub struct sunKeyLongPair {
    pub key: &'static str,
    pub set: sunLongSetFn,
}

pub fn sunCheckAndSetLongArgs(
    mem: &mut Option<Box<dyn Any>>,
    argidx: &mut i32,
    argv: &[String],
    offset: usize,
    testpairs: &[sunKeyLongPair],
    numpairs: i32,
    arg_used: &mut sunbooleantype,
    failedarg: &mut i32,
) -> i32 {
    for j in 0..numpairs {
        *arg_used = SUNFALSE;
        if argv[*argidx as usize][offset..] == *testpairs[j as usize].key {
            *argidx += 1;
            let iarg: i64 = crate::sundials_utils::atol(&argv[*argidx as usize]);
            let retval = (testpairs[j as usize].set)(mem, iarg);
            if retval != SUN_SUCCESS {
                *failedarg = j;
                return retval;
            }
            *arg_used = SUNTRUE;
            return SUN_SUCCESS;
        }
    }
    SUN_SUCCESS
}

/* utilities for pair int/sunrealtype "set" routines */
pub type sunIntRealSetFn = fn(mem: &mut Option<Box<dyn Any>>, arg1: i32, arg2: sunrealtype) -> i32;

#[derive(Clone, Copy)]
pub struct sunKeyIntRealPair {
    pub key: &'static str,
    pub set: sunIntRealSetFn,
}

pub fn sunCheckAndSetIntRealArgs(
    mem: &mut Option<Box<dyn Any>>,
    argidx: &mut i32,
    argv: &[String],
    offset: usize,
    testpairs: &[sunKeyIntRealPair],
    numpairs: i32,
    arg_used: &mut sunbooleantype,
    failedarg: &mut i32,
) -> i32 {
    for j in 0..numpairs {
        *arg_used = SUNFALSE;
        if argv[*argidx as usize][offset..] == *testpairs[j as usize].key {
            *argidx += 1;
            let iarg: i32 = crate::sundials_utils::atoi(&argv[*argidx as usize]);
            *argidx += 1;
            let rarg: sunrealtype = SUNStrToReal(&argv[*argidx as usize]);
            let retval = (testpairs[j as usize].set)(mem, iarg, rarg);
            if retval != SUN_SUCCESS {
                *failedarg = j;
                return retval;
            }
            *arg_used = SUNTRUE;
            return SUN_SUCCESS;
        }
    }
    SUN_SUCCESS
}

/* utilities for triplet int/sunrealtype/sunrealtype "set" routines */
pub type sunIntRealRealSetFn =
    fn(mem: &mut Option<Box<dyn Any>>, arg1: i32, arg2: sunrealtype, arg3: sunrealtype) -> i32;

#[derive(Clone, Copy)]
pub struct sunKeyIntRealRealPair {
    pub key: &'static str,
    pub set: sunIntRealRealSetFn,
}

pub fn sunCheckAndSetIntRealRealArgs(
    mem: &mut Option<Box<dyn Any>>,
    argidx: &mut i32,
    argv: &[String],
    offset: usize,
    testpairs: &[sunKeyIntRealRealPair],
    numpairs: i32,
    arg_used: &mut sunbooleantype,
    failedarg: &mut i32,
) -> i32 {
    for j in 0..numpairs {
        *arg_used = SUNFALSE;
        if argv[*argidx as usize][offset..] == *testpairs[j as usize].key {
            *argidx += 1;
            let iarg: i32 = crate::sundials_utils::atoi(&argv[*argidx as usize]);
            *argidx += 1;
            let rarg1: sunrealtype = SUNStrToReal(&argv[*argidx as usize]);
            *argidx += 1;
            let rarg2: sunrealtype = SUNStrToReal(&argv[*argidx as usize]);
            let retval = (testpairs[j as usize].set)(mem, iarg, rarg1, rarg2);
            if retval != SUN_SUCCESS {
                *failedarg = j;
                return retval;
            }
            *arg_used = SUNTRUE;
            return SUN_SUCCESS;
        }
    }
    SUN_SUCCESS
}

/* utilities for pair int/long int "set" routines */
pub type sunIntLongSetFn = fn(mem: &mut Option<Box<dyn Any>>, arg1: i32, arg2: i64) -> i32;

#[derive(Clone, Copy)]
pub struct sunKeyIntLongPair {
    pub key: &'static str,
    pub set: sunIntLongSetFn,
}

pub fn sunCheckAndSetIntLongArgs(
    mem: &mut Option<Box<dyn Any>>,
    argidx: &mut i32,
    argv: &[String],
    offset: usize,
    testpairs: &[sunKeyIntLongPair],
    numpairs: i32,
    arg_used: &mut sunbooleantype,
    failedarg: &mut i32,
) -> i32 {
    for j in 0..numpairs {
        *arg_used = SUNFALSE;
        if argv[*argidx as usize][offset..] == *testpairs[j as usize].key {
            *argidx += 1;
            let iarg: i32 = crate::sundials_utils::atoi(&argv[*argidx as usize]);
            *argidx += 1;
            let large: i64 = crate::sundials_utils::atol(&argv[*argidx as usize]);
            let retval = (testpairs[j as usize].set)(mem, iarg, large);
            if retval != SUN_SUCCESS {
                *failedarg = j;
                return retval;
            }
            *arg_used = SUNTRUE;
            return SUN_SUCCESS;
        }
    }
    SUN_SUCCESS
}

/* utilities for sunrealtype "set" routines */
pub type sunRealSetFn = fn(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32;

#[derive(Clone, Copy)]
pub struct sunKeyRealPair {
    pub key: &'static str,
    pub set: sunRealSetFn,
}

pub fn sunCheckAndSetRealArgs(
    mem: &mut Option<Box<dyn Any>>,
    argidx: &mut i32,
    argv: &[String],
    offset: usize,
    testpairs: &[sunKeyRealPair],
    numpairs: i32,
    arg_used: &mut sunbooleantype,
    failedarg: &mut i32,
) -> i32 {
    for j in 0..numpairs {
        *arg_used = SUNFALSE;
        if argv[*argidx as usize][offset..] == *testpairs[j as usize].key {
            *argidx += 1;
            let rarg: sunrealtype = SUNStrToReal(&argv[*argidx as usize]);
            let retval = (testpairs[j as usize].set)(mem, rarg);
            if retval != SUN_SUCCESS {
                *failedarg = j;
                return retval;
            }
            *arg_used = SUNTRUE;
            return SUN_SUCCESS;
        }
    }
    SUN_SUCCESS
}

/* utilities for pair-of-sunrealtype "set" routines */
pub type sunTwoRealSetFn =
    fn(mem: &mut Option<Box<dyn Any>>, arg1: sunrealtype, arg2: sunrealtype) -> i32;

#[derive(Clone, Copy)]
pub struct sunKeyTwoRealPair {
    pub key: &'static str,
    pub set: sunTwoRealSetFn,
}

pub fn sunCheckAndSetTwoRealArgs(
    mem: &mut Option<Box<dyn Any>>,
    argidx: &mut i32,
    argv: &[String],
    offset: usize,
    testpairs: &[sunKeyTwoRealPair],
    numpairs: i32,
    arg_used: &mut sunbooleantype,
    failedarg: &mut i32,
) -> i32 {
    for j in 0..numpairs {
        *arg_used = SUNFALSE;
        if argv[*argidx as usize][offset..] == *testpairs[j as usize].key {
            *argidx += 1;
            let rarg1: sunrealtype = SUNStrToReal(&argv[*argidx as usize]);
            *argidx += 1;
            let rarg2: sunrealtype = SUNStrToReal(&argv[*argidx as usize]);
            let retval = (testpairs[j as usize].set)(mem, rarg1, rarg2);
            if retval != SUN_SUCCESS {
                *failedarg = j;
                return retval;
            }
            *arg_used = SUNTRUE;
            return SUN_SUCCESS;
        }
    }
    SUN_SUCCESS
}

/* utilities for char* "set" routines */
pub type sunCharSetFn = fn(mem: &mut Option<Box<dyn Any>>, arg: &str) -> i32;

#[derive(Clone, Copy)]
pub struct sunKeyCharPair {
    pub key: &'static str,
    pub set: sunCharSetFn,
}

pub fn sunCheckAndSetCharArgs(
    mem: &mut Option<Box<dyn Any>>,
    argidx: &mut i32,
    argv: &[String],
    offset: usize,
    testpairs: &[sunKeyCharPair],
    numpairs: i32,
    arg_used: &mut sunbooleantype,
    failedarg: &mut i32,
) -> i32 {
    for j in 0..numpairs {
        *arg_used = SUNFALSE;
        if argv[*argidx as usize][offset..] == *testpairs[j as usize].key {
            *argidx += 1;
            let retval = (testpairs[j as usize].set)(mem, argv[*argidx as usize].as_str());
            if retval != SUN_SUCCESS {
                *failedarg = j;
                return retval;
            }
            *arg_used = SUNTRUE;
            return SUN_SUCCESS;
        }
    }
    SUN_SUCCESS
}

/* utilities for pair-of-char* "set" routines */
pub type sunTwoCharSetFn = fn(mem: &mut Option<Box<dyn Any>>, arg1: &str, arg2: &str) -> i32;

#[derive(Clone, Copy)]
pub struct sunKeyTwoCharPair {
    pub key: &'static str,
    pub set: sunTwoCharSetFn,
}

pub fn sunCheckAndSetTwoCharArgs(
    mem: &mut Option<Box<dyn Any>>,
    argidx: &mut i32,
    argv: &[String],
    offset: usize,
    testpairs: &[sunKeyTwoCharPair],
    numpairs: i32,
    arg_used: &mut sunbooleantype,
    failedarg: &mut i32,
) -> i32 {
    for j in 0..numpairs {
        *arg_used = SUNFALSE;
        if argv[*argidx as usize][offset..] == *testpairs[j as usize].key {
            /* C reads argv[*argidx + 1] and argv[*argidx + 2] before
             * advancing, then advances by 2 before checking retval */
            let retval = (testpairs[j as usize].set)(
                mem,
                argv[(*argidx + 1) as usize].as_str(),
                argv[(*argidx + 2) as usize].as_str(),
            );
            *argidx += 2;
            if retval != SUN_SUCCESS {
                *failedarg = j;
                return retval;
            }
            *arg_used = SUNTRUE;
            return SUN_SUCCESS;
        }
    }
    SUN_SUCCESS
}

/* utilities for action "set" routines */
pub type sunActionSetFn = fn(mem: &mut Option<Box<dyn Any>>) -> i32;

#[derive(Clone, Copy)]
pub struct sunKeyActionPair {
    pub key: &'static str,
    pub set: sunActionSetFn,
}

pub fn sunCheckAndSetActionArgs(
    mem: &mut Option<Box<dyn Any>>,
    argidx: &mut i32,
    argv: &[String],
    offset: usize,
    testpairs: &[sunKeyActionPair],
    numpairs: i32,
    arg_used: &mut sunbooleantype,
    failedarg: &mut i32,
) -> i32 {
    for j in 0..numpairs {
        *arg_used = SUNFALSE;
        if argv[*argidx as usize][offset..] == *testpairs[j as usize].key {
            let retval = (testpairs[j as usize].set)(mem);
            if retval != SUN_SUCCESS {
                *failedarg = j;
                return retval;
            }
            *arg_used = SUNTRUE;
            return SUN_SUCCESS;
        }
    }
    SUN_SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Target {
        ints: Vec<i32>,
        longs: Vec<i64>,
        reals: Vec<sunrealtype>,
        strs: Vec<String>,
        actions: i32,
    }

    fn target(mem: &mut Option<Box<dyn Any>>) -> &mut Target {
        mem.as_mut()
            .expect("mem token present")
            .downcast_mut::<Target>()
            .expect("Target content")
    }

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    fn set_int(mem: &mut Option<Box<dyn Any>>, arg: i32) -> i32 {
        target(mem).ints.push(arg);
        SUN_SUCCESS
    }

    fn set_int_fail(_mem: &mut Option<Box<dyn Any>>, _arg: i32) -> i32 {
        -1
    }

    fn set_two_int(mem: &mut Option<Box<dyn Any>>, arg1: i32, arg2: i32) -> i32 {
        target(mem).ints.push(arg1);
        target(mem).ints.push(arg2);
        SUN_SUCCESS
    }

    fn set_long(mem: &mut Option<Box<dyn Any>>, arg: i64) -> i32 {
        target(mem).longs.push(arg);
        SUN_SUCCESS
    }

    fn set_real(mem: &mut Option<Box<dyn Any>>, arg: sunrealtype) -> i32 {
        target(mem).reals.push(arg);
        SUN_SUCCESS
    }

    fn set_two_char(mem: &mut Option<Box<dyn Any>>, arg1: &str, arg2: &str) -> i32 {
        target(mem).strs.push(arg1.to_string());
        target(mem).strs.push(arg2.to_string());
        SUN_SUCCESS
    }

    fn set_action(mem: &mut Option<Box<dyn Any>>) -> i32 {
        target(mem).actions += 1;
        SUN_SUCCESS
    }

    #[test]
    fn int_match_consumes_one_value() {
        let mut mem: Option<Box<dyn Any>> = Some(Box::new(Target::default()));
        let argv = args(&["prog", "cvode.max_order", "3"]);
        let pairs = [sunKeyIntPair { key: "max_order", set: set_int }];
        let mut idx: i32 = 1;
        let mut used = SUNFALSE;
        let mut failed: i32 = -1;
        /* offset 6 strips "cvode." */
        let retval =
            sunCheckAndSetIntArgs(&mut mem, &mut idx, &argv, 6, &pairs, 1, &mut used, &mut failed);
        assert_eq!(retval, SUN_SUCCESS);
        assert_eq!(used, SUNTRUE);
        assert_eq!(idx, 2);
        assert_eq!(failed, -1);
        assert_eq!(target(&mut mem).ints, vec![3]);
    }

    #[test]
    fn no_match_leaves_index_and_sets_arg_used_false() {
        let mut mem: Option<Box<dyn Any>> = Some(Box::new(Target::default()));
        /* "--cvode.max_order" must NOT match with offset for "cvode." */
        let argv = args(&["prog", "--cvode.max_order", "3"]);
        let pairs = [sunKeyIntPair { key: "max_order", set: set_int }];
        let mut idx: i32 = 1;
        let mut used = SUNTRUE;
        let mut failed: i32 = -1;
        let retval =
            sunCheckAndSetIntArgs(&mut mem, &mut idx, &argv, 6, &pairs, 1, &mut used, &mut failed);
        assert_eq!(retval, SUN_SUCCESS);
        assert_eq!(used, SUNFALSE);
        assert_eq!(idx, 1);
        assert!(target(&mut mem).ints.is_empty());
    }

    #[test]
    fn zero_pairs_leaves_arg_used_untouched() {
        let mut mem: Option<Box<dyn Any>> = Some(Box::new(Target::default()));
        let argv = args(&["prog", "cvode.max_order", "3"]);
        let pairs: [sunKeyIntPair; 0] = [];
        let mut idx: i32 = 1;
        let mut used = SUNTRUE;
        let mut failed: i32 = -1;
        let retval =
            sunCheckAndSetIntArgs(&mut mem, &mut idx, &argv, 6, &pairs, 0, &mut used, &mut failed);
        assert_eq!(retval, SUN_SUCCESS);
        assert_eq!(used, SUNTRUE); /* untouched, as in C */
        assert_eq!(idx, 1);
    }

    #[test]
    fn failing_setter_reports_failedarg_after_consuming_value() {
        let mut mem: Option<Box<dyn Any>> = Some(Box::new(Target::default()));
        let argv = args(&["prog", "cvode.max_order", "3"]);
        let pairs = [
            sunKeyIntPair { key: "other", set: set_int },
            sunKeyIntPair { key: "max_order", set: set_int_fail },
        ];
        let mut idx: i32 = 1;
        let mut used = SUNFALSE;
        let mut failed: i32 = -1;
        let retval =
            sunCheckAndSetIntArgs(&mut mem, &mut idx, &argv, 6, &pairs, 2, &mut used, &mut failed);
        assert_eq!(retval, -1);
        assert_eq!(used, SUNFALSE);
        assert_eq!(failed, 1);
        assert_eq!(idx, 2); /* value index already consumed */
    }

    #[test]
    fn two_int_consumes_two_values() {
        let mut mem: Option<Box<dyn Any>> = Some(Box::new(Target::default()));
        let argv = args(&["prog", "ida.orders", "2", "5"]);
        let pairs = [sunKeyTwoIntPair { key: "orders", set: set_two_int }];
        let mut idx: i32 = 1;
        let mut used = SUNFALSE;
        let mut failed: i32 = -1;
        let retval = sunCheckAndSetTwoIntArgs(&mut mem, &mut idx, &argv, 4, &pairs, 1, &mut used,
                                              &mut failed);
        assert_eq!(retval, SUN_SUCCESS);
        assert_eq!(used, SUNTRUE);
        assert_eq!(idx, 3);
        assert_eq!(target(&mut mem).ints, vec![2, 5]);
    }

    #[test]
    fn long_and_real_parsing() {
        let mut mem: Option<Box<dyn Any>> = Some(Box::new(Target::default()));
        let argv = args(&["prog", "kinsol.num_max_iters", "12345678901", "kinsol.tol", "1.5e-8"]);
        let long_pairs = [sunKeyLongPair { key: "num_max_iters", set: set_long }];
        let real_pairs = [sunKeyRealPair { key: "tol", set: set_real }];
        let mut idx: i32 = 1;
        let mut used = SUNFALSE;
        let mut failed: i32 = -1;
        let retval = sunCheckAndSetLongArgs(&mut mem, &mut idx, &argv, 7, &long_pairs, 1, &mut used,
                                            &mut failed);
        assert_eq!(retval, SUN_SUCCESS);
        assert_eq!(idx, 2);
        idx += 1;
        let retval = sunCheckAndSetRealArgs(&mut mem, &mut idx, &argv, 7, &real_pairs, 1, &mut used,
                                            &mut failed);
        assert_eq!(retval, SUN_SUCCESS);
        assert_eq!(idx, 4);
        assert_eq!(target(&mut mem).longs, vec![12345678901]);
        assert_eq!(target(&mut mem).reals, vec![1.5e-8]);
    }

    #[test]
    fn two_char_reads_ahead_then_advances_two() {
        let mut mem: Option<Box<dyn Any>> = Some(Box::new(Target::default()));
        let argv = args(&["prog", "arkode.tables", "ERK_1", "DIRK_2"]);
        let pairs = [sunKeyTwoCharPair { key: "tables", set: set_two_char }];
        let mut idx: i32 = 1;
        let mut used = SUNFALSE;
        let mut failed: i32 = -1;
        let retval = sunCheckAndSetTwoCharArgs(&mut mem, &mut idx, &argv, 7, &pairs, 1, &mut used,
                                               &mut failed);
        assert_eq!(retval, SUN_SUCCESS);
        assert_eq!(used, SUNTRUE);
        assert_eq!(idx, 3);
        assert_eq!(target(&mut mem).strs, vec!["ERK_1".to_string(), "DIRK_2".to_string()]);
    }

    #[test]
    fn action_consumes_no_values() {
        let mut mem: Option<Box<dyn Any>> = Some(Box::new(Target::default()));
        let argv = args(&["prog", "cvode.no_min_step"]);
        let pairs = [sunKeyActionPair { key: "no_min_step", set: set_action }];
        let mut idx: i32 = 1;
        let mut used = SUNFALSE;
        let mut failed: i32 = -1;
        let retval = sunCheckAndSetActionArgs(&mut mem, &mut idx, &argv, 6, &pairs, 1, &mut used,
                                              &mut failed);
        assert_eq!(retval, SUN_SUCCESS);
        assert_eq!(used, SUNTRUE);
        assert_eq!(idx, 1);
        assert_eq!(target(&mut mem).actions, 1);
    }

    #[test]
    fn atoi_semantics_unparsable_gives_zero() {
        let mut mem: Option<Box<dyn Any>> = Some(Box::new(Target::default()));
        let argv = args(&["prog", "cvode.max_order", "abc"]);
        let pairs = [sunKeyIntPair { key: "max_order", set: set_int }];
        let mut idx: i32 = 1;
        let mut used = SUNFALSE;
        let mut failed: i32 = -1;
        let retval =
            sunCheckAndSetIntArgs(&mut mem, &mut idx, &argv, 6, &pairs, 1, &mut used, &mut failed);
        assert_eq!(retval, SUN_SUCCESS);
        assert_eq!(target(&mut mem).ints, vec![0]);
    }
}
