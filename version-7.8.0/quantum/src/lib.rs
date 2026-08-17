//! Quantum mechanics on a grid: bound states, propagation, and
//! scattering in 1, 2 and 3 dimensions.
//!
//! The core is the Rust port of the 1-D quantum machinery from SolveIt.
//! It does the two things a 1-D quantum solver is for:
//!
//! * **Bound states** — discretise `H = -hbar^2/2m d^2/dx^2 + V(x)` on a
//!   grid and diagonalise it, giving eigenvalues and eigenfunctions for
//!   an *arbitrary* potential;
//! * **Time evolution** — propagate a wavepacket with a Crank–Nicolson
//!   step, which is unitary for any time step, and read off observables
//!   including transmission and reflection.
//!
//! [`nash`] adds a second propagator: the **Bessel-expanded
//! split-operator** scheme from the original C++, which is explicit and
//! matrix-free where Crank–Nicolson solves a tridiagonal system, and
//! periodic where [`qm1d`] has walls. It is the one piece of original
//! numerical work in SolveIt and is ported faithfully, Lie splitting and
//! all.
//!
//! Everything is built on `special_functions`: the complex tridiagonal
//! solver for the propagator, and the Jacobi eigensolver for the bound
//! states. Both of those are clean-room replacements for the
//! licence-encumbered routines in the original C++ — see
//! `CLEANROOM_PROVENANCE.md`.
//!
//! Beyond 1-D, the crate carries:
//!
//! * [`qm2d`] / [`qm3d`] — 2-D and 3-D grids: ADI propagation with
//!   Strang-composed Cayley factors (exactly unitary at any step) and
//!   Lanczos bound states; `qm3d` adds marginal densities;
//! * [`transfer`] — fixed-energy scattering by transfer matrix:
//!   `T(E)`, `R(E)`, flux balance, and a measured conditioning number;
//! * [`absorber`] — the absorbing boundary, designed rather than tuned
//!   (reflection and leakage computed exactly from the same transfer
//!   matrix);
//! * [`isosurface`] — marching tetrahedra, for rendering 3-D densities.
//!
//! # Units
//!
//! `hbar` and the mass are explicit fields rather than being fixed to 1,
//! because getting a wrong answer in the "obvious" units is a classic
//! way to lose an afternoon. They default to 1, which is what almost
//! every textbook problem wants.
//!
//! # Boundary conditions (the 1-D modules)
//!
//! Dirichlet for [`qm1d`]: the wavefunction is pinned to zero just
//! outside the grid,
//! so the box is an infinite well. For bound states that is physical
//! whenever the domain is wide enough that the state has decayed. For
//! scattering it means the walls **reflect** — the domain must be long
//! enough that nothing reaches them within the simulated time, and
//! [`Wavefunction::edge_probability`] exists so you can check that
//! assumption instead of assuming it.

#![forbid(unsafe_code)]
#![deny(warnings)]
#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]

pub mod absorber;
pub mod isosurface;
pub mod nash;
pub mod qm1d;
pub mod qm2d;
pub mod qm3d;
pub mod transfer;
