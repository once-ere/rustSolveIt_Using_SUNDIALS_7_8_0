//! `ida_rs` — pure-Rust port of SUNDIALS 7.8.0 `src/ida`, part of
//! **SUNDIALS_7_8_Rust_port_for_AppleSilicon_macos**.
//!
//! **Platform scope.** Portable Rust (`std` only, no `unsafe`, no FFI, no
//! `cfg(target_os)`/`cfg(target_arch)`); builds and unit-tests anywhere. The
//! project's byte-identical-output guarantee against the upstream C reference
//! examples was established only on **macOS / Apple Silicon (arm64)**,
//! because `sin`, `cos`, `exp`, `ln` and the inverse/hyperbolic functions
//! come from the host libm — `pow` alone was made host-independent. (`sqrt`,
//! `mul_add`, `ceil`, `round`, `abs` and `copysign` are IEEE-exact, hence
//! portable.) See `README.md` § "Platform scope".
//!
//! The accepted-deviation notes below were likewise argued unobservable on
//! that platform.

#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

/* =================================================================
 * Accepted deviations (verification pass, Phase 5)
 *
 * Verified divergences from the release-C reference build that are
 * deliberate and unobservable on every path a valid serial example
 * takes. Mirror these in `idas_rs`; do not "fix" them back toward C.
 *
 * A. Diagnostic source coordinates. Every `IDAProcessError` call site
 *    passes Rust `line!()`/`file!()` where C passes `__LINE__`/
 *    `__FILE__`; the func argument is C's exact `__func__` string (or
 *    the forwarded `fname` parameter) at all 168 sites. Those three
 *    values only reach `sunCombineFileAndLine` -> the logger *scope*
 *    field on the WARNING (`MSG_INACTIVE_ROOTS`) and ERROR paths, so
 *    the divergence is confined to the
 *    `[WARNING][rank 0][<file>:<line>][<func>]` prefix of a diagnostic
 *    line. None of the 37 IDA reference `.out` files contains such a
 *    line. Rust source coordinates cannot reproduce C ones; the pattern
 *    is uniform across all ported crates (~2200 sites), so if
 *    byte-identical diagnostic text is ever required it must be done
 *    workspace-wide with per-site upstream path/line literals, never
 *    piecemeal in one module.
 *
 * B. Callback-window state locals. `IDARootfind` moves the mutated
 *    rootfinding arrays (`ida_glo/ghi/grout/iroots/rootdir/gactive`)
 *    into locals for the whole Illinois search and writes them back at
 *    a single exit; `IDARcheck1/2/3` do the same around single
 *    statements. This is the locked granular-borrow pattern (no RefCell
 *    borrow may be held across the user `g` call). The written-back
 *    values equal C's per-return-path writes on every path, including
 *    `IDA_RTFUNC_FAIL`, so no output differs. The only divergence is a
 *    getter invoked *re-entrantly from inside* `ida_gfun`
 *    (`IDAGetRootInfo` reading `ida_iroots`), which sees an emptied Vec
 *    and panics where C read live, partially-updated, meaningless
 *    values. Mid-callback getters are not supported SUNDIALS usage.
 *    Do NOT half-restore a subset of the arrays: that would convert a
 *    loud panic into a silently clobbered re-entrant write.
 *
 * C. Freed-but-not-NULLed C fields. `IDASetId(mem, NULL)`
 *    (`ida_io.c:524-560`) destroys `ida_id` and clears
 *    `ida_idMallocDone` but leaves `ida_id` dangling;
 *    `IDASetConstraints(mem, NULL)` (`ida_io.c:565-600`) destroys
 *    `ida_constraints` with no subsequent assignment. Every later
 *    truth-test of those fields (`IDAInitialSetup`'s constraint mask
 *    and id/suppressalg checks, `IDASolve`'s step constraint check,
 *    `IDACalcIC`'s `IDA_YA_YDP_INIT` id check, and the
 *    `idaLsDenseDQJac`/`idaLsBandDQJac`/`IBBDDQJac` sign-adjustment
 *    guards) therefore dereferences freed memory in C. The port drops
 *    the handle (`Option::take` -> `None`), so those guards take the
 *    "absent" branch instead. Sibling of the "C UB -> deterministic
 *    behavior" class: the C path is use-after-free, so no reference
 *    example can depend on it, and the Rust result matches the
 *    documented SUNDIALS semantics.
 * =================================================================*/

/* ida modules (one per upstream C file) */
pub mod ida;
pub mod ida_bbdpre;
pub mod ida_cli;
pub mod ida_ic;
pub mod ida_impl;
pub mod ida_io;
pub mod ida_ls;
pub mod ida_nls;

/* Re-export every shared module from sundials_core (workspace rule) */
pub use sundials_core::nvector_serial;
pub use sundials_core::sunadaptcontroller_imexgus;
pub use sundials_core::sunadaptcontroller_mrihtol;
pub use sundials_core::sunadaptcontroller_soderlind;
pub use sundials_core::sundials_adaptcontroller;
pub use sundials_core::sundials_band;
pub use sundials_core::sundials_cli;
pub use sundials_core::sundials_context;
pub use sundials_core::sundials_dense;
pub use sundials_core::sundials_direct;
pub use sundials_core::sundials_errors;
pub use sundials_core::sundials_futils;
pub use sundials_core::sundials_hashmap;
pub use sundials_core::sundials_iterative;
pub use sundials_core::sundials_linearsolver;
pub use sundials_core::sundials_logger;
pub use sundials_core::sundials_libm;
pub use sundials_core::sundials_math;
pub use sundials_core::sundials_sparse_lu;
pub use sundials_core::sundials_matrix;
pub use sundials_core::sundials_memory;
pub use sundials_core::sundials_nonlinearsolver;
pub use sundials_core::sundials_nvector;
pub use sundials_core::sundials_nvector_senswrapper;
pub use sundials_core::sundials_profiler;
pub use sundials_core::sundials_system_memory;
pub use sundials_core::sundials_types;
pub use sundials_core::sundials_utils;
pub use sundials_core::sundials_version;
pub use sundials_core::sunlinsol_band;
pub use sundials_core::sunlinsol_dense;
pub use sundials_core::sunlinsol_klu;
pub use sundials_core::sunlinsol_pcg;
pub use sundials_core::sunlinsol_spbcgs;
pub use sundials_core::sunlinsol_spfgmr;
pub use sundials_core::sunlinsol_spgmr;
pub use sundials_core::sunlinsol_sptfqmr;
pub use sundials_core::sunmatrix_band;
pub use sundials_core::sunmatrix_dense;
pub use sundials_core::sunmatrix_sparse;
pub use sundials_core::sunnonlinsol_auto;
pub use sundials_core::sunnonlinsol_fixedpoint;
pub use sundials_core::sunnonlinsol_newton;
pub use sundials_core::sunstl_vector;

/* Flat prelude so examples can `use ida_rs::*;` */
pub mod prelude {
    pub use crate::ida::*;
    pub use crate::ida_bbdpre::*;
    pub use crate::ida_cli::*;
    pub use crate::ida_ic::*;
    pub use crate::ida_impl::*;
    pub use crate::ida_io::*;
    pub use crate::ida_ls::*;
    pub use crate::ida_nls::*;
    pub use sundials_core::nvector_serial::*;
    pub use sundials_core::sundials_context::*;
    pub use sundials_core::sundials_errors::*;
    pub use sundials_core::sundials_iterative::*;
    pub use sundials_core::sundials_linearsolver::*;
    pub use sundials_core::sundials_libm::SunMath;
    pub use sundials_core::sundials_math::*;
    pub use sundials_core::sundials_matrix::*;
    pub use sundials_core::sundials_nonlinearsolver::*;
    pub use sundials_core::sundials_nvector::*;
    pub use sundials_core::sundials_types::*;
    pub use sundials_core::sundials_utils::*;
    pub use sundials_core::sunlinsol_band::*;
    pub use sundials_core::sunlinsol_dense::*;
    pub use sundials_core::sunlinsol_klu::*;
    pub use sundials_core::sunlinsol_pcg::*;
    pub use sundials_core::sunlinsol_spbcgs::*;
    pub use sundials_core::sunlinsol_spfgmr::*;
    pub use sundials_core::sunlinsol_spgmr::*;
    pub use sundials_core::sunlinsol_sptfqmr::*;
    pub use sundials_core::sunmatrix_band::*;
    pub use sundials_core::sunmatrix_dense::*;
    pub use sundials_core::sunmatrix_sparse::*;
    pub use sundials_core::sunnonlinsol_auto::*;
    pub use sundials_core::sunnonlinsol_fixedpoint::*;
    pub use sundials_core::sunnonlinsol_newton::*;
}
