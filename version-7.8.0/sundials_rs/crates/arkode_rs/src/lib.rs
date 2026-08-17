//! `arkode_rs` — pure-Rust port of SUNDIALS 7.8.0 `src/arkode`, part of
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

/* arkode modules (one per upstream C file) */
pub mod arkode;
pub mod arkode_adapt;
pub mod arkode_arkstep;
pub mod arkode_arkstep_io;
pub mod arkode_arkstep_nls;
pub mod arkode_bandpre;
pub mod arkode_bbdpre;
pub mod arkode_butcher;
pub mod arkode_butcher_dirk;
pub mod arkode_butcher_erk;
pub mod arkode_cli;
pub mod arkode_erkstep;
pub mod arkode_erkstep_io;
pub mod arkode_forcingstep;
pub mod arkode_impl;
pub mod arkode_interp;
pub mod arkode_io;
pub mod arkode_ls;
pub mod arkode_lsrkstep;
pub mod arkode_lsrkstep_io;
pub mod arkode_mri_tables;
pub mod arkode_mristep;
pub mod arkode_mristep_controller;
pub mod arkode_mristep_io;
pub mod arkode_mristep_nls;
pub mod arkode_relaxation;
pub mod arkode_root;
pub mod arkode_splittingstep;
pub mod arkode_splittingstep_coefficients;
pub mod arkode_sprk;
pub mod arkode_sprkstep;
pub mod arkode_sprkstep_io;
pub mod arkode_sunstepper;
pub mod arkode_user_controller;

/* Re-export every shared module from sundials_core (workspace rule) */
pub use sundials_core::nvector_serial;
pub use sundials_core::sunadaptcontroller_imexgus;
pub use sundials_core::sunadaptcontroller_mrihtol;
pub use sundials_core::sunadaptcontroller_soderlind;
pub use sundials_core::sunadjointcheckpointscheme_fixed;
pub use sundials_core::sundatanode_inmem;
pub use sundials_core::sundials_adaptcontroller;
pub use sundials_core::sundials_adjointcheckpointscheme;
pub use sundials_core::sundials_adjointstepper;
pub use sundials_core::sundials_band;
pub use sundials_core::sundials_cli;
pub use sundials_core::sundials_context;
pub use sundials_core::sundials_datanode;
pub use sundials_core::sundials_dense;
pub use sundials_core::sundials_direct;
pub use sundials_core::sundials_domeigestimator;
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
pub use sundials_core::sundials_stepper;
pub use sundials_core::sundials_system_memory;
pub use sundials_core::sundials_types;
pub use sundials_core::sundials_utils;
pub use sundials_core::sundials_version;
pub use sundials_core::sundomeigest_arnoldi;
pub use sundials_core::sundomeigest_power;
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

/* Flat prelude so examples can `use arkode_rs::*;` */
pub mod prelude {
    pub use crate::arkode::*;
    pub use crate::arkode_adapt::*;
    pub use crate::arkode_arkstep::*;
    pub use crate::arkode_arkstep_io::*;
    pub use crate::arkode_arkstep_nls::*;
    pub use crate::arkode_bandpre::*;
    pub use crate::arkode_bbdpre::*;
    pub use crate::arkode_butcher::*;
    pub use crate::arkode_butcher_dirk::*;
    pub use crate::arkode_butcher_erk::*;
    pub use crate::arkode_cli::*;
    pub use crate::arkode_erkstep::*;
    pub use crate::arkode_erkstep_io::*;
    pub use crate::arkode_forcingstep::*;
    pub use crate::arkode_impl::*;
    pub use crate::arkode_interp::*;
    pub use crate::arkode_io::*;
    pub use crate::arkode_ls::*;
    pub use crate::arkode_lsrkstep::*;
    pub use crate::arkode_lsrkstep_io::*;
    pub use crate::arkode_mri_tables::*;
    pub use crate::arkode_mristep::*;
    pub use crate::arkode_mristep_controller::*;
    pub use crate::arkode_mristep_io::*;
    pub use crate::arkode_mristep_nls::*;
    pub use crate::arkode_relaxation::*;
    pub use crate::arkode_root::*;
    pub use crate::arkode_splittingstep::*;
    pub use crate::arkode_splittingstep_coefficients::*;
    pub use crate::arkode_sprk::*;
    pub use crate::arkode_sprkstep::*;
    pub use crate::arkode_sprkstep_io::*;
    pub use crate::arkode_sunstepper::*;
    pub use crate::arkode_user_controller::*;
    pub use sundials_core::nvector_serial::*;
    pub use sundials_core::sundials_adaptcontroller::*;
    pub use sundials_core::sundials_context::*;
    pub use sundials_core::sundials_domeigestimator::*;
    pub use sundials_core::sundials_errors::*;
    pub use sundials_core::sundials_iterative::*;
    pub use sundials_core::sundials_linearsolver::*;
    pub use sundials_core::sundials_libm::SunMath;
    pub use sundials_core::sundials_math::*;
    pub use sundials_core::sundials_matrix::*;
    pub use sundials_core::sundials_nonlinearsolver::*;
    pub use sundials_core::sundials_nvector::*;
    pub use sundials_core::sundials_stepper::*;
    pub use sundials_core::sundials_types::*;
    pub use sundials_core::sundials_utils::*;
    pub use sundials_core::sunadaptcontroller_imexgus::*;
    pub use sundials_core::sunadaptcontroller_mrihtol::*;
    pub use sundials_core::sunadaptcontroller_soderlind::*;
    pub use sundials_core::sundomeigest_arnoldi::*;
    pub use sundials_core::sundomeigest_power::*;
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

    /* Two names are defined identically by more than one upstream impl
       header (`MSG_NLS_INIT_FAIL` in arkode_arkstep_impl.h and
       arkode_mristep_impl.h; `SIX` in arkode_impl.h's interpolation section
       and arkode_lsrkstep_impl.h). Each module keeps its own definition, as
       in C; these explicit re-exports pick one so the flat prelude resolves
       the name instead of leaving it ambiguous. Values are identical. */
    pub use crate::arkode_arkstep::MSG_NLS_INIT_FAIL;
    pub use crate::arkode_interp::SIX;
}
