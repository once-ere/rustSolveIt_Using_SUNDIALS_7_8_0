//! `sundials_core` — shared library of
//! **SUNDIALS_7_8_Rust_port_for_Linux_on_ubuntu**, a pure-Rust
//! line-by-line port of SUNDIALS 7.8.0.
//!
//! # Platform scope
//!
//! The code is portable: `std` only, no `unsafe`, no FFI, no external crates
//! and no `cfg(target_os)`/`cfg(target_arch)` anywhere. It builds
//! warning-free and passes its unit tests on any target Rust supports.
//!
//! Unlike the macOS/arm64 and Linux/glibc repositories this tree inherits its
//! translation from, **no elementary function here resolves to the host C
//! library.** [`sundials_libm`] implements `exp`, `log`, `pow`, `expm1`,
//! `log1p`, `sin`, `cos`, `atan`, `asin`, `acos`, `sinh`, `cosh` and `acosh`
//! in pure Rust, and every call site in the library and in the translated
//! examples goes through it (spelled `x.sun_sin()`, `x.sun_exp()`, …). The
//! only `f64` methods still used are `sqrt`, `mul_add`, `abs`, `ceil`,
//! `round` and `copysign` — all IEEE-754 specified, correctly rounded, and
//! identical on every target.
//!
//! The practical consequence is that the numerical output of this port is a
//! function of its own source only: it does not move when the host glibc
//! version moves. `tools/libm_differential.sh` measures, function by
//! function, how the pure-Rust routines relate to the host libm on the
//! machine the port is built on; `LIBM.md` records the result.
//!
//! The reference platform for the *example* results is Ubuntu 26.04 LTS on
//! x86-64 (glibc 2.43, gcc 15.2.0, rustc 1.96.1). See `README.md`.

#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

pub mod sunadjointcheckpointscheme_fixed;
pub mod sundatanode_inmem;
pub mod sundials_adjointcheckpointscheme;
pub mod sundials_adjointstepper;
pub mod sundials_context;
pub mod sundials_datanode;
pub mod sundials_domeigestimator;
pub mod sundials_errors;
pub mod sundials_futils;
pub mod sundials_hashmap;
pub mod sundials_iterative;
pub mod sundials_linearsolver;
pub mod sundials_logger;
pub mod nvector_manyvector;
pub mod nvector_serial;
pub mod sunadaptcontroller_imexgus;
pub mod sunadaptcontroller_mrihtol;
pub mod sunadaptcontroller_soderlind;
pub mod sundials_adaptcontroller;
pub mod sundials_band;
pub mod sundials_cli;
pub mod sundials_dense;
pub mod sundials_direct;
pub mod sundials_libm;
pub mod sundials_math;
pub mod sundials_sparse_lu;
pub mod sundials_stepper;
pub mod sundomeigest_arnoldi;
pub mod sundomeigest_power;
pub mod sunlinsol_band;
pub mod sunlinsol_dense;
pub mod sunlinsol_klu;
pub mod sunlinsol_pcg;
pub mod sunlinsol_spbcgs;
pub mod sunlinsol_spfgmr;
pub mod sunlinsol_spgmr;
pub mod sunlinsol_sptfqmr;
pub mod sundials_matrix;
pub mod sundials_memory;
pub mod sundials_nonlinearsolver;
pub mod sundials_nvector;
pub mod sundials_nvector_senswrapper;
pub mod sundials_profiler;
pub mod sundials_system_memory;
pub mod sundials_types;
pub mod sundials_utils;
pub mod sundials_version;
pub mod sunmatrix_band;
pub mod sunnonlinsol_auto;
pub mod sunnonlinsol_fixedpoint;
pub mod sunnonlinsol_newton;
pub mod sunmatrix_dense;
pub mod sunmatrix_sparse;
pub mod sunstl_vector;
