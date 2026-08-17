//! `physical_object` — the unique union of the three legacy simulator types
//! (`PointParticle`, `RigidBody`, `RigidBody3D`) as a single pure-Rust
//! library whose only numerical-integration backend is the local
//! `sundials_rs` workspace (pure-Rust SUNDIALS 7.7.0).
//!
//! Guarantees: zero `unsafe`, zero external crate dependencies,
//! zero warnings.
#![forbid(unsafe_code)]
#![deny(warnings)]
#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]

pub mod linalg;
pub mod boundary;
pub mod physical_object;
pub mod system;
pub mod integrate;
pub mod collide;

pub use crate::boundary::{analytic_inertia_tensor, Boundary, Sdf};
pub use crate::collide::Contact;
pub use crate::integrate::{Method, RunReport, Snapshot};
pub use crate::linalg::{Mat3, Quat, Vec3};
pub use crate::system::PhysicalObjectSystem;
// NOTE: the struct `physical_object` shares its name with its module, so
// it cannot also be re-exported at the crate root; import it as
// `use physical_object::physical_object::physical_object;`.
