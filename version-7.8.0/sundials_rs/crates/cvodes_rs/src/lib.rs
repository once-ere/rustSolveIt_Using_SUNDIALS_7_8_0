//! `cvodes_rs` — pure-Rust port of SUNDIALS 7.8.0 `src/cvodes`, part of
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

/* cvodes modules (one per upstream C file) */
pub mod cvodea;
pub mod cvodea_io;
pub mod cvodes;
pub mod cvodes_bandpre;
pub mod cvodes_bbdpre;
pub mod cvodes_cli;
pub mod cvodes_diag;
pub mod cvodes_impl;
pub mod cvodes_io;
pub mod cvodes_ls;
pub mod cvodes_nls;
pub mod cvodes_nls_sim;
pub mod cvodes_nls_stg;
pub mod cvodes_nls_stg1;
pub mod cvodes_proj;
pub mod cvodes_resize;

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
    pub use crate::cvodea::*;
    pub use crate::cvodea_io::*;
    pub use crate::cvodes::*;
    pub use crate::cvodes_bandpre::*;
    pub use crate::cvodes_bbdpre::*;
    pub use crate::cvodes_cli::*;
    pub use crate::cvodes_diag::*;
    pub use crate::cvodes_impl::*;
    pub use crate::cvodes_io::*;
    pub use crate::cvodes_ls::*;
    pub use crate::cvodes_nls::*;
    pub use crate::cvodes_nls_sim::*;
    pub use crate::cvodes_nls_stg::*;
    pub use crate::cvodes_nls_stg1::*;
    pub use crate::cvodes_proj::*;
    pub use crate::cvodes_resize::*;
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
