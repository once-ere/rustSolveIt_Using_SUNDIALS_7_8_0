//! `pub struct physical_object` — the **unique union** of the three legacy
//! types. Every data member and property of `PointParticle`, `RigidBody`
//! and `RigidBody3D` appears exactly once:
//!
//! | field | donor(s) | note |
//! |---|---|---|
//! | `id` | PointParticle | |
//! | `mass`, `inverse_mass` | all / RigidBody3D | coupled: `m <= 0 → inv = 0` |
//! | `charge` | RigidBody, RigidBody3D | |
//! | `position` | all | |
//! | `orientation` | RigidBody | setter renormalizes |
//! | `momentum` | RigidBody, RigidBody3D | canonical; PointParticle's `velocity` is the derived property `velocity()` |
//! | `angular_momentum` | RigidBody, RigidBody3D | spin; PointParticle's orbital method is `orbital_angular_momentum(com)` |
//! | `inertia_tensor`, `inverse_inertia_tensor` | RigidBody + RigidBody3D | body frame, coupled setters |
//! | `magnetic_moment_tensor` | RigidBody | torque = (R M Rᵀ)·B |
//! | `boundary` | RigidBody (`dyn` SDF) + RigidBody3D (enum) | unified `Boundary` enum implementing [`Sdf`] |
//!
//! The legacy per-object Euler integrators are intentionally **not**
//! carried over: all time integration is delegated to `sundials_rs`
//! (see [`crate::integrate`]).

use crate::boundary::{analytic_inertia_tensor, Boundary, Sdf};
use crate::linalg::{Mat3, Quat, Vec3};

/// The unique union of `PointParticle`, `RigidBody` and `RigidBody3D`.
#[derive(Clone, Debug, PartialEq)]
pub struct physical_object {
    pub(crate) id: usize,
    pub(crate) mass: f64,
    pub(crate) inverse_mass: f64,
    pub(crate) charge: f64,
    pub(crate) position: Vec3,
    pub(crate) orientation: Quat,
    pub(crate) momentum: Vec3,
    pub(crate) angular_momentum: Vec3,
    pub(crate) inertia_tensor: Mat3,
    pub(crate) inverse_inertia_tensor: Mat3,
    pub(crate) magnetic_moment_tensor: Mat3,
    pub(crate) boundary: Boundary,
    pub(crate) restitution: f64,
}

fn inverse_of_mass(mass: f64) -> f64 {
    if mass > 0.0 {
        1.0 / mass
    } else {
        0.0
    }
}

fn inverse_of_tensor(t: &Mat3) -> Mat3 {
    // Singular tensors (e.g. a point particle) invert to the zero
    // tensor: the body simply cannot rotate, mirroring the
    // `inverse_mass = 0` convention for static bodies.
    t.try_inverse().unwrap_or_else(Mat3::zeros)
}

impl physical_object {
    // ----------------------------------------------------------------
    // Constructors — one per donor type.
    // ----------------------------------------------------------------

    /// `PointParticle::new` equivalent: a point mass with position and
    /// velocity (stored canonically as momentum `p = m v`).
    pub fn new_point(id: usize, mass: f64, position: Vec3, velocity: Vec3) -> Self {
        Self {
            id,
            mass,
            inverse_mass: inverse_of_mass(mass),
            charge: 0.0,
            position,
            orientation: Quat::identity(),
            momentum: mass * velocity,
            angular_momentum: Vec3::zeros(),
            inertia_tensor: Mat3::zeros(),
            inverse_inertia_tensor: Mat3::zeros(),
            magnetic_moment_tensor: Mat3::zeros(),
            boundary: Boundary::Point,
            restitution: 1.0,
        }
    }

    /// `RigidBody::new` equivalent: explicit orientation and tensors;
    /// initial momenta computed exactly as the donor did
    /// (`p = m v`, `L = (R I Rᵀ) w`).
    #[allow(clippy::too_many_arguments)]
    pub fn new_rigid(
        id: usize,
        mass: f64,
        charge: f64,
        position: Vec3,
        orientation: Quat,
        linear_velocity: Vec3,
        angular_velocity: Vec3,
        inertia_tensor: Mat3,
        magnetic_moment_tensor: Mat3,
        boundary: Boundary,
    ) -> Self {
        let orientation = orientation.normalize();
        let momentum = mass * linear_velocity;
        let inverse_inertia_tensor = inverse_of_tensor(&inertia_tensor);
        let rot = orientation.to_rotation_matrix();
        let world_inertia = rot * inertia_tensor * rot.transpose();
        let angular_momentum = world_inertia * angular_velocity;
        Self {
            id,
            mass,
            inverse_mass: inverse_of_mass(mass),
            charge,
            position,
            orientation,
            momentum,
            angular_momentum,
            inertia_tensor,
            inverse_inertia_tensor,
            magnetic_moment_tensor,
            boundary,
            restitution: 1.0,
        }
    }

    /// `RigidBody3D::new` equivalent: inertia tensor auto-computed from
    /// the boundary shape, identity orientation, `L = I w` (diagonal).
    pub fn new_from_shape(
        id: usize,
        mass: f64,
        charge: f64,
        position: Vec3,
        linear_velocity: Vec3,
        angular_velocity: Vec3,
        boundary: Boundary,
    ) -> Self {
        let inertia_tensor = analytic_inertia_tensor(mass, &boundary);
        let inverse_inertia_tensor = inertia_tensor.inverse_diagonal();
        let momentum = mass * linear_velocity;
        let angular_momentum = inertia_tensor * angular_velocity;
        Self {
            id,
            mass,
            inverse_mass: inverse_of_mass(mass),
            charge,
            position,
            orientation: Quat::identity(),
            momentum,
            angular_momentum,
            inertia_tensor,
            inverse_inertia_tensor,
            magnetic_moment_tensor: Mat3::zeros(),
            boundary,
            restitution: 1.0,
        }
    }

    // ----------------------------------------------------------------
    // get / set for every field.
    // ----------------------------------------------------------------

    pub fn get_id(&self) -> usize {
        self.id
    }

    pub fn set_id(&mut self, id: usize) {
        self.id = id;
    }

    pub fn get_mass(&self) -> f64 {
        self.mass
    }

    /// Sets the mass and keeps `inverse_mass` consistent
    /// (`m <= 0 → inverse_mass = 0`, the static-body convention).
    pub fn set_mass(&mut self, mass: f64) {
        self.mass = mass;
        self.inverse_mass = inverse_of_mass(mass);
    }

    pub fn get_inverse_mass(&self) -> f64 {
        self.inverse_mass
    }

    /// Sets the inverse mass and back-computes `mass`
    /// (`inverse_mass = 0 → mass = 0`).
    pub fn set_inverse_mass(&mut self, inverse_mass: f64) {
        self.inverse_mass = inverse_mass;
        self.mass = inverse_of_mass(inverse_mass);
    }

    pub fn get_charge(&self) -> f64 {
        self.charge
    }

    pub fn set_charge(&mut self, charge: f64) {
        self.charge = charge;
    }

    pub fn get_position(&self) -> Vec3 {
        self.position
    }

    pub fn set_position(&mut self, position: Vec3) {
        self.position = position;
    }

    pub fn get_orientation(&self) -> Quat {
        self.orientation
    }

    /// Sets the orientation, renormalizing to a unit quaternion.
    pub fn set_orientation(&mut self, orientation: Quat) {
        self.orientation = orientation.normalize();
    }

    pub fn get_momentum(&self) -> Vec3 {
        self.momentum
    }

    pub fn set_momentum(&mut self, momentum: Vec3) {
        self.momentum = momentum;
    }

    pub fn get_angular_momentum(&self) -> Vec3 {
        self.angular_momentum
    }

    pub fn set_angular_momentum(&mut self, angular_momentum: Vec3) {
        self.angular_momentum = angular_momentum;
    }

    pub fn get_inertia_tensor(&self) -> Mat3 {
        self.inertia_tensor
    }

    /// Sets the body-frame inertia tensor and keeps its inverse
    /// consistent (singular → zero inverse: non-rotating body).
    pub fn set_inertia_tensor(&mut self, inertia_tensor: Mat3) {
        self.inertia_tensor = inertia_tensor;
        self.inverse_inertia_tensor = inverse_of_tensor(&inertia_tensor);
    }

    pub fn get_inverse_inertia_tensor(&self) -> Mat3 {
        self.inverse_inertia_tensor
    }

    /// Sets the inverse inertia tensor directly and back-computes the
    /// forward tensor (singular → zero tensor).
    pub fn set_inverse_inertia_tensor(&mut self, inverse_inertia_tensor: Mat3) {
        self.inverse_inertia_tensor = inverse_inertia_tensor;
        self.inertia_tensor = inverse_of_tensor(&inverse_inertia_tensor);
    }

    pub fn get_magnetic_moment_tensor(&self) -> Mat3 {
        self.magnetic_moment_tensor
    }

    pub fn set_magnetic_moment_tensor(&mut self, magnetic_moment_tensor: Mat3) {
        self.magnetic_moment_tensor = magnetic_moment_tensor;
    }

    /// Coefficient of restitution `e ∈ [0, 1]` used by the collision
    /// response (1 = perfectly elastic, 0 = perfectly plastic;
    /// default 1). A colliding pair uses `min(e_i, e_j)`.
    pub fn get_restitution(&self) -> f64 {
        self.restitution
    }

    /// Sets the coefficient of restitution, clamped into `[0, 1]`.
    pub fn set_restitution(&mut self, restitution: f64) {
        self.restitution = restitution.clamp(0.0, 1.0);
    }

    pub fn get_boundary(&self) -> Boundary {
        self.boundary
    }

    /// Sets the boundary shape. Deliberately does **not** touch the
    /// inertia tensor (RigidBody-style users may have hand-set tensors);
    /// call [`Self::recompute_inertia_from_boundary`] for the
    /// RigidBody3D behavior.
    pub fn set_boundary(&mut self, boundary: Boundary) {
        self.boundary = boundary;
    }

    // ----------------------------------------------------------------
    // Derived properties with get / set (PointParticle's `velocity`
    // member and the donors' velocity accessors live here).
    // ----------------------------------------------------------------

    /// Current linear velocity `v = p * m⁻¹` (safe for `m = 0`).
    pub fn get_velocity(&self) -> Vec3 {
        self.momentum * self.inverse_mass
    }

    /// Sets the linear velocity by writing the canonical momentum
    /// `p = m v`.
    pub fn set_velocity(&mut self, velocity: Vec3) {
        self.momentum = self.mass * velocity;
    }

    /// Legacy `RigidBody` / `RigidBody3D` name for [`Self::get_velocity`].
    pub fn linear_velocity(&self) -> Vec3 {
        self.get_velocity()
    }

    /// Current world-space angular velocity `w = (R I⁻¹ Rᵀ) L`
    /// (general RigidBody form; reduces to RigidBody3D's diagonal form
    /// at identity orientation).
    pub fn get_angular_velocity(&self) -> Vec3 {
        self.world_inverse_inertia() * self.angular_momentum
    }

    /// World-space inverse inertia tensor `R I⁻¹ Rᵀ` — the quantity the
    /// contact solver needs to turn an impulse at a point into an
    /// angular-velocity change. (Exactly the expression previously
    /// inlined in [`Self::get_angular_velocity`].)
    pub fn world_inverse_inertia(&self) -> Mat3 {
        let r = self.orientation.to_rotation_matrix();
        r * self.inverse_inertia_tensor * r.transpose()
    }

    /// Sets the world-space angular velocity by writing the canonical
    /// angular momentum `L = (R I Rᵀ) w`.
    pub fn set_angular_velocity(&mut self, angular_velocity: Vec3) {
        let r = self.orientation.to_rotation_matrix();
        let world_inertia = r * self.inertia_tensor * r.transpose();
        self.angular_momentum = world_inertia * angular_velocity;
    }

    /// Legacy name for [`Self::get_angular_velocity`].
    pub fn angular_velocity(&self) -> Vec3 {
        self.get_angular_velocity()
    }

    // ----------------------------------------------------------------
    // Union methods (PointParticle observables).
    // ----------------------------------------------------------------

    /// Linear momentum `p = m v` (PointParticle method; now the stored
    /// canonical field).
    pub fn momentum(&self) -> Vec3 {
        self.momentum
    }

    /// Orbital angular momentum relative to a center-of-mass point:
    /// `L_orb = r_rel × p` (PointParticle's `angular_momentum(com)`,
    /// renamed to avoid colliding with the stored spin field).
    pub fn orbital_angular_momentum(&self, center_of_mass: Vec3) -> Vec3 {
        let r_rel = self.position - center_of_mass;
        r_rel.cross(self.momentum())
    }

    /// Total angular momentum about a point: orbital + spin.
    pub fn total_angular_momentum(&self, center_of_mass: Vec3) -> Vec3 {
        self.orbital_angular_momentum(center_of_mass) + self.angular_momentum
    }

    /// Laplace-Runge-Lenz vector relative to a moving center-of-mass
    /// point: `A = p × L − m k r_hat`, `k = G · M_total` (verbatim
    /// PointParticle port, including the `r = 0` guard).
    pub fn laplace_vector(&self, center_of_mass: Vec3, k: f64) -> Vec3 {
        let p = self.momentum();
        let l = self.orbital_angular_momentum(center_of_mass);

        let r_rel = self.position - center_of_mass;
        let r_norm = r_rel.norm();

        // Prevent division by zero if the object lands perfectly on the
        // center of mass.
        let r_hat = if r_norm > 0.0 {
            r_rel / r_norm
        } else {
            Vec3::zeros()
        };

        p.cross(l) - (self.mass * k * r_hat)
    }

    // ----------------------------------------------------------------
    // Union methods (RigidBody).
    // ----------------------------------------------------------------

    /// Total kinetic energy (translational + rotational):
    /// `½ m |v|² + ½ w·L`.
    pub fn kinetic_energy(&self) -> f64 {
        let v = self.linear_velocity();
        let w = self.angular_velocity();
        let translational = 0.5 * self.mass * v.norm_squared();
        let rotational = 0.5 * w.dot(self.angular_momentum);
        translational + rotational
    }

    /// Transforms a point from world space to the body's local frame.
    pub fn to_local_space(&self, world_point: &Vec3) -> Vec3 {
        self.orientation.inverse().rotate(*world_point - self.position)
    }

    /// Transforms a point from the body's local frame to world space.
    pub fn to_world_space(&self, local_point: &Vec3) -> Vec3 {
        self.orientation.rotate(*local_point) + self.position
    }

    /// Signed distance from a local point to the boundary surface
    /// (RigidBody's `boundary.signed_distance`, via the unified enum).
    pub fn signed_distance(&self, local_point: &Vec3) -> f64 {
        self.boundary.signed_distance(local_point)
    }

    /// Local surface normal via central differences (RigidBody default).
    pub fn surface_normal(&self, local_point: &Vec3) -> Vec3 {
        self.boundary.surface_normal(local_point)
    }

    // ----------------------------------------------------------------
    // Union methods (RigidBody3D).
    // ----------------------------------------------------------------

    /// Recomputes the inertia tensor (and inverse) analytically from
    /// the current boundary and mass (RigidBody3D `new` behavior).
    pub fn recompute_inertia_from_boundary(&mut self) {
        self.inertia_tensor = analytic_inertia_tensor(self.mass, &self.boundary);
        self.inverse_inertia_tensor = self.inertia_tensor.inverse_diagonal();
    }

    /// Integrates this object's state forward by `dt` under constant
    /// external force and torque.
    ///
    /// The legacy `RigidBody`/`RigidBody3D` semi-implicit/explicit Euler
    /// bodies are **not** used: this delegates to the sundials_rs CVODE
    /// driver (see [`crate::integrate::propagate_single`]).
    pub fn integrate(&mut self, force: &Vec3, torque: &Vec3, dt: f64) -> Result<(), String> {
        crate::integrate::propagate_single(self, *force, *torque, dt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coupled_mass_setters() {
        let mut o = physical_object::new_point(0, 2.0, Vec3::zeros(), Vec3::zeros());
        assert_eq!(o.get_inverse_mass(), 0.5);
        o.set_mass(4.0);
        assert_eq!(o.get_inverse_mass(), 0.25);
        o.set_inverse_mass(0.0);
        assert_eq!(o.get_mass(), 0.0);
        o.set_mass(-1.0);
        assert_eq!(o.get_inverse_mass(), 0.0);
    }

    #[test]
    fn velocity_momentum_roundtrip() {
        let mut o = physical_object::new_point(1, 2.0, Vec3::zeros(), Vec3::new(1.0, -2.0, 3.0));
        assert_eq!(o.get_momentum(), Vec3::new(2.0, -4.0, 6.0));
        assert_eq!(o.get_velocity(), Vec3::new(1.0, -2.0, 3.0));
        o.set_velocity(Vec3::new(0.5, 0.0, 0.0));
        assert_eq!(o.get_momentum(), Vec3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn point_particle_observables_match_donor() {
        // Donor formulas computed by hand for mass 2, r = (1,0,0), v = (0,3,0):
        // p = (0,6,0); L about origin = r x p = (0,0,6);
        // A = p x L - m k r_hat with k = 5: (36,0,0) - (10,0,0) = (26,0,0).
        let o = physical_object::new_point(7, 2.0, Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 3.0, 0.0));
        assert_eq!(o.momentum(), Vec3::new(0.0, 6.0, 0.0));
        assert_eq!(o.orbital_angular_momentum(Vec3::zeros()), Vec3::new(0.0, 0.0, 6.0));
        assert_eq!(o.laplace_vector(Vec3::zeros(), 5.0), Vec3::new(26.0, 0.0, 0.0));
        // r = 0 guard
        let c = physical_object::new_point(8, 1.0, Vec3::zeros(), Vec3::zeros());
        assert_eq!(c.laplace_vector(Vec3::zeros(), 5.0), Vec3::zeros());
    }

    #[test]
    fn rigid_constructor_matches_donor_init() {
        // RigidBody::new: momentum = m v; L = (R I R^T) w. At identity
        // orientation with diagonal I this is L = I w.
        let i = Mat3::from_diagonal(Vec3::new(2.0, 3.0, 4.0));
        let o = physical_object::new_rigid(
            0,
            2.0,
            -1.5,
            Vec3::zeros(),
            Quat::identity(),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
            i,
            Mat3::zeros(),
            Boundary::Sphere { radius: 1.0 },
        );
        assert_eq!(o.get_momentum(), Vec3::new(2.0, 0.0, 0.0));
        assert_eq!(o.get_angular_momentum(), Vec3::new(0.0, 6.0, 0.0));
        assert_eq!(o.get_inverse_inertia_tensor().0[1][1], 1.0 / 3.0);
        // kinetic energy: 0.5*2*1 + 0.5*w.L = 1 + 0.5*2*6 = 7
        assert!((o.kinetic_energy() - 7.0).abs() < 1e-12);
    }

    #[test]
    fn shape_constructor_matches_rigidbody3d() {
        // RigidBody3D main(): sphere r=0.5, m=2 → I = 0.4*2*0.25 = 0.2 diag;
        // w = (0,2,0) → L = (0,0.4,0); v = p/m round-trips.
        let o = physical_object::new_from_shape(
            0,
            2.0,
            -1.5,
            Vec3::new(0.0, 10.0, 0.0),
            Vec3::new(1.0, 0.0, -0.5),
            Vec3::new(0.0, 2.0, 0.0),
            Boundary::Sphere { radius: 0.5 },
        );
        assert!((o.get_inertia_tensor().0[0][0] - 0.2).abs() < 1e-15);
        assert!((o.get_angular_momentum().y - 0.4).abs() < 1e-15);
        assert_eq!(o.linear_velocity(), Vec3::new(1.0, 0.0, -0.5));
        let w = o.angular_velocity();
        assert!((w.y - 2.0).abs() < 1e-12 && w.x.abs() < 1e-12);
    }

    #[test]
    fn local_world_space_roundtrip() {
        let q = Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), 0.7);
        let mut o = physical_object::new_point(0, 1.0, Vec3::new(1.0, 2.0, 3.0), Vec3::zeros());
        o.set_orientation(q);
        let p = Vec3::new(-0.3, 0.8, 2.0);
        let back = o.to_local_space(&o.to_world_space(&p));
        assert!((back - p).norm() < 1e-13);
    }

    #[test]
    fn tensor_setters_stay_coupled() {
        let mut o = physical_object::new_point(0, 1.0, Vec3::zeros(), Vec3::zeros());
        let i = Mat3([[2.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 4.0]]);
        o.set_inertia_tensor(i);
        let prod = o.get_inertia_tensor() * o.get_inverse_inertia_tensor();
        assert!((prod.0[0][0] - 1.0).abs() < 1e-12 && prod.0[0][1].abs() < 1e-12);
        // singular → zero inverse
        o.set_inertia_tensor(Mat3::zeros());
        assert_eq!(o.get_inverse_inertia_tensor(), Mat3::zeros());
    }
}
