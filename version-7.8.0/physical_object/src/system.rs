//! `PhysicalObjectSystem` — the union successor of the legacy
//! `GravitationalSystem` (`particles` → `objects`; the `SOFTENING`
//! constant becomes the `softening` field, default 1e-6), extended with
//! the uniform fields the rigid-body donors imply (gravity, E, B,
//! per-object external force/torque) and solver settings.
//!
//! The legacy `step()` (hand-rolled velocity Verlet) is replaced by the
//! sundials_rs drivers in [`crate::integrate`].

use crate::collide::Contact;
use crate::integrate::Method;
use crate::linalg::Vec3;
use crate::physical_object::physical_object;

/// Number of state variables per object:
/// `[pos(3) | momentum(3) | quat(4: w,x,y,z) | angular_momentum(3)]`.
pub const VARS_PER_OBJECT: usize = 13;

/// A collection of [`physical_object`]s plus global interaction
/// parameters and solver settings.
#[derive(Clone, Debug)]
pub struct PhysicalObjectSystem {
    pub objects: Vec<physical_object>,
    /// Newtonian gravitational constant `G` (legacy `g_constant`).
    pub g_constant: f64,
    /// Plummer softening length for pairwise gravity (legacy constant 1e-6).
    pub softening: f64,
    /// Uniform gravitational acceleration field (e.g. `[0, -9.81, 0]`).
    pub uniform_gravity: Vec3,
    /// Uniform electric field `E`; force `q E`.
    pub e_field: Vec3,
    /// Uniform magnetic field `B`; force `q v × B`, torque `(R M Rᵀ) B`.
    pub b_field: Vec3,
    /// Constant external force per object (indexed like `objects`).
    pub external_forces: Vec<Vec3>,
    /// Constant external torque per object (indexed like `objects`).
    pub external_torques: Vec<Vec3>,
    /// Relative tolerance for the CVODE path.
    pub rtol: f64,
    /// Absolute tolerance for the CVODE path.
    pub atol: f64,
    /// Integration method (all methods are sundials_rs solvers).
    pub method: Method,
    /// Current simulation time.
    pub time: f64,
    /// Rigid-body collision detection master switch (default `true`).
    /// When enabled and at least one collidable pair exists, the
    /// integrator arms sundials rootfinding on the pairwise signed
    /// separations and resolves impulses at each detected contact.
    pub collide_enabled: bool,
    /// Relative normal speeds below this are treated as resting contact
    /// (restitution forced to 0) to prevent settling jitter.
    pub restitution_threshold: f64,
    /// Penetration slop tolerated before positional projection pushes
    /// overlapping bodies apart.
    pub contact_slop: f64,
    /// Contacts recorded by the most recent `integrate::run`/`step`
    /// (cleared at the start of each; capped at
    /// [`crate::collide::CONTACTS_CAP`], oldest dropped).
    pub contacts: Vec<Contact>,
    /// Rigid holonomic constraints (`CONSTRAIN`). Non-empty means the
    /// system is a DAE, and `Method::Ida` is the only integrator that can
    /// honour it — see `constrain.rs` and ARCHITECTURE §3.9.
    pub constraints: crate::constrain::ConstraintSet,
    /// Running total of resolved collision impulses this session.
    pub collision_count: u64,
}

impl Default for PhysicalObjectSystem {
    fn default() -> Self {
        Self::new(Vec::new(), 1.0)
    }
}

impl PhysicalObjectSystem {
    /// `GravitationalSystem::new` equivalent.
    pub fn new(objects: Vec<physical_object>, g_constant: f64) -> Self {
        let n = objects.len();
        Self {
            objects,
            g_constant,
            softening: 1e-6,
            uniform_gravity: Vec3::zeros(),
            e_field: Vec3::zeros(),
            b_field: Vec3::zeros(),
            external_forces: vec![Vec3::zeros(); n],
            external_torques: vec![Vec3::zeros(); n],
            rtol: 1.0e-10,
            atol: 1.0e-12,
            method: Method::Adams,
            time: 0.0,
            collide_enabled: true,
            restitution_threshold: 1e-3,
            contact_slop: 1e-9,
            contacts: Vec::new(),
            constraints: crate::constrain::ConstraintSet::default(),
            collision_count: 0,
        }
    }

    /// Adds an object (growing the external force/torque tables) and
    /// returns its index.
    pub fn add_object(&mut self, obj: physical_object) -> usize {
        self.objects.push(obj);
        self.external_forces.push(Vec3::zeros());
        self.external_torques.push(Vec3::zeros());
        self.objects.len() - 1
    }

    /// Removes the object at `index` along with its external tables.
    pub fn remove_object(&mut self, index: usize) -> Option<physical_object> {
        if index >= self.objects.len() {
            return None;
        }
        self.external_forces.remove(index);
        self.external_torques.remove(index);
        Some(self.objects.remove(index))
    }

    /// Total combined mass (`GravitationalSystem::total_mass`).
    pub fn total_mass(&self) -> f64 {
        self.objects.iter().map(|o| o.mass).sum()
    }

    /// Shared center-of-mass position (`GravitationalSystem::center_of_mass`).
    pub fn center_of_mass(&self) -> Vec3 {
        let total_mass = self.total_mass();
        if total_mass == 0.0 {
            return Vec3::zeros();
        }
        let mut mass_pos_sum = Vec3::zeros();
        for o in &self.objects {
            mass_pos_sum += o.mass * o.position;
        }
        mass_pos_sum / total_mass
    }

    /// Collective softened gravitational accelerations
    /// (`GravitationalSystem::compute_accelerations`, verbatim math).
    // An N-body double loop indexes `accelerations[i]` while reading
    // `objects[j]`; clippy's suggested iterator form would need the two
    // collections zipped, which obscures the pairwise structure this
    // code exists to express.
    #[allow(clippy::needless_range_loop)]
    pub fn compute_accelerations(&self) -> Vec<Vec3> {
        let n = self.objects.len();
        let mut accelerations = vec![Vec3::zeros(); n];
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let r_vec = self.objects[j].position - self.objects[i].position;
                let dist_sq = r_vec.norm_squared() + self.softening * self.softening;
                let dist = dist_sq.sqrt();
                accelerations[i] += (self.g_constant * self.objects[j].mass / (dist_sq * dist)) * r_vec;
            }
        }
        accelerations
    }

    /// Total linear momentum.
    pub fn total_momentum(&self) -> Vec3 {
        let mut p = Vec3::zeros();
        for o in &self.objects {
            p += o.momentum;
        }
        p
    }

    /// Total angular momentum about `about`: orbital + spin of every object.
    pub fn total_angular_momentum(&self, about: Vec3) -> Vec3 {
        let mut l = Vec3::zeros();
        for o in &self.objects {
            l += o.total_angular_momentum(about);
        }
        l
    }

    /// Total mechanical energy: kinetic (translational + rotational)
    /// plus softened pairwise gravitational potential plus the uniform
    /// gravity and electric-field potentials. (The magnetic torque
    /// coupling `(R M Rᵀ) B` is not derived from a potential, so energy
    /// is only conserved when that tensor is zero.)
    pub fn total_energy(&self) -> f64 {
        let mut e = 0.0;
        for o in &self.objects {
            e += o.kinetic_energy();
            e -= o.mass * self.uniform_gravity.dot(o.position);
            e -= o.charge * self.e_field.dot(o.position);
        }
        let n = self.objects.len();
        for i in 0..n {
            for j in (i + 1)..n {
                let r_vec = self.objects[j].position - self.objects[i].position;
                let dist = (r_vec.norm_squared() + self.softening * self.softening).sqrt();
                e -= self.g_constant * self.objects[i].mass * self.objects[j].mass / dist;
            }
        }
        e
    }

    /// Laplace-Runge-Lenz vector of object `i` about the system center
    /// of mass with `k = G · M_total` (the legacy main-loop recipe).
    pub fn laplace_vector(&self, i: usize) -> Option<Vec3> {
        let o = self.objects.get(i)?;
        let com = self.center_of_mass();
        let k = self.g_constant * self.total_mass();
        Some(o.laplace_vector(com, k))
    }

    // ----------------------------------------------------------------
    // State packing for the sundials solvers.
    // ----------------------------------------------------------------

    /// Number of packed state variables (`13 N`).
    pub fn state_len(&self) -> usize {
        VARS_PER_OBJECT * self.objects.len()
    }

    /// Packs all object states into a flat vector with the layout
    /// `[pos(3) | momentum(3) | quat(w,x,y,z) | L(3)]` per object.
    pub fn pack_state(&self) -> Vec<f64> {
        let mut y = Vec::with_capacity(self.state_len());
        for o in &self.objects {
            y.extend_from_slice(&[o.position.x, o.position.y, o.position.z]);
            y.extend_from_slice(&[o.momentum.x, o.momentum.y, o.momentum.z]);
            y.extend_from_slice(&[o.orientation.w, o.orientation.x, o.orientation.y, o.orientation.z]);
            y.extend_from_slice(&[
                o.angular_momentum.x,
                o.angular_momentum.y,
                o.angular_momentum.z,
            ]);
        }
        y
    }

    /// Writes a packed state vector back into the objects
    /// (orientation quaternions are renormalized).
    pub fn unpack_state(&mut self, y: &[f64]) {
        assert_eq!(y.len(), self.state_len(), "state length mismatch");
        for (i, o) in self.objects.iter_mut().enumerate() {
            let b = VARS_PER_OBJECT * i;
            o.position = Vec3::new(y[b], y[b + 1], y[b + 2]);
            o.momentum = Vec3::new(y[b + 3], y[b + 4], y[b + 5]);
            o.orientation =
                crate::linalg::Quat::new(y[b + 6], y[b + 7], y[b + 8], y[b + 9]).normalize();
            o.angular_momentum = Vec3::new(y[b + 10], y[b + 11], y[b + 12]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::Quat;

    #[test]
    fn com_and_total_mass_match_donor() {
        let p1 = physical_object::new_point(1, 1.0, Vec3::new(1.0, 0.0, 0.0), Vec3::zeros());
        let p2 = physical_object::new_point(2, 3.0, Vec3::new(-1.0, 0.0, 0.0), Vec3::zeros());
        let s = PhysicalObjectSystem::new(vec![p1, p2], 1.0);
        assert_eq!(s.total_mass(), 4.0);
        assert_eq!(s.center_of_mass(), Vec3::new(-0.5, 0.0, 0.0));
        let empty = PhysicalObjectSystem::new(vec![], 1.0);
        assert_eq!(empty.center_of_mass(), Vec3::zeros());
    }

    #[test]
    fn accelerations_match_donor_formula() {
        // Two unit masses 2 apart, G = 1, softening eps: a = 2/(4+eps^2)^{3/2} ≈ 0.25.
        let p1 = physical_object::new_point(1, 1.0, Vec3::new(1.0, 0.0, 0.0), Vec3::zeros());
        let p2 = physical_object::new_point(2, 1.0, Vec3::new(-1.0, 0.0, 0.0), Vec3::zeros());
        let s = PhysicalObjectSystem::new(vec![p1, p2], 1.0);
        let a = s.compute_accelerations();
        assert!((a[0].x + 0.25).abs() < 1e-9);
        assert!((a[1].x - 0.25).abs() < 1e-9);
    }

    #[test]
    fn pack_unpack_roundtrip() {
        let mut o = physical_object::new_from_shape(
            0,
            2.0,
            -1.5,
            Vec3::new(0.0, 10.0, 0.0),
            Vec3::new(1.0, 0.0, -0.5),
            Vec3::new(0.0, 2.0, 0.0),
            crate::boundary::Boundary::Sphere { radius: 0.5 },
        );
        o.set_orientation(Quat::from_axis_angle(Vec3::new(1.0, 1.0, 0.0), 0.4));
        let mut s = PhysicalObjectSystem::new(vec![o], 1.0);
        let y = s.pack_state();
        assert_eq!(y.len(), VARS_PER_OBJECT);
        let before = s.objects[0].clone();
        s.unpack_state(&y);
        let after = &s.objects[0];
        assert!((before.get_position() - after.get_position()).norm() < 1e-15);
        assert!((before.get_momentum() - after.get_momentum()).norm() < 1e-15);
        assert!((before.get_angular_momentum() - after.get_angular_momentum()).norm() < 1e-15);
        assert!((before.get_orientation().norm() - 1.0).abs() < 1e-15);
    }

    #[test]
    fn add_remove_keep_tables_aligned() {
        let mut s = PhysicalObjectSystem::default();
        let i = s.add_object(physical_object::new_point(0, 1.0, Vec3::zeros(), Vec3::zeros()));
        s.add_object(physical_object::new_point(1, 2.0, Vec3::zeros(), Vec3::zeros()));
        assert_eq!(i, 0);
        assert_eq!(s.external_forces.len(), 2);
        s.remove_object(0);
        assert_eq!(s.objects.len(), 1);
        assert_eq!(s.external_forces.len(), 1);
        assert_eq!(s.objects[0].get_id(), 1);
    }
}
