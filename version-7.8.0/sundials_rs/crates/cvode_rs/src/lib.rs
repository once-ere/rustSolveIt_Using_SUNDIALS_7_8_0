//! `cvode_rs` — pure-Rust port of SUNDIALS 7.8.0 `src/cvode`, part of
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

#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

/* cvode modules (one per upstream C file) */
pub mod cvode;
pub mod cvode_bandpre;
pub mod cvode_bbdpre;
pub mod cvode_cli;
pub mod cvode_diag;
pub mod cvode_fused_stubs;
pub mod cvode_impl;
pub mod cvode_io;
pub mod cvode_ls;
pub mod cvode_nls;
pub mod cvode_proj;
pub mod cvode_resize;

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

/* Flat prelude so examples can `use cvode_rs::*;` */
pub mod prelude {
    pub use crate::cvode::*;
    pub use crate::cvode_bandpre::*;
    pub use crate::cvode_bbdpre::*;
    pub use crate::cvode_cli::*;
    pub use crate::cvode_diag::*;
    pub use crate::cvode_fused_stubs::*;
    pub use crate::cvode_impl::*;
    pub use crate::cvode_io::*;
    pub use crate::cvode_ls::*;
    pub use crate::cvode_nls::*;
    pub use crate::cvode_proj::*;
    pub use crate::cvode_resize::*;
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
