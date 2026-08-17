//! Rigid holonomic constraints, and the algebra the DAE solvers need.
//!
//! A constraint is a geometric relation the motion must satisfy *exactly*
//! — a rod of fixed length between two bodies, say — rather than a stiff
//! spring that only approximately enforces it. Enforcing it exactly turns
//! the equations of motion from an ODE into a **differential-algebraic
//! equation** (DAE), which is what `ida_rs`/`idas_rs` are for.
//!
//! # The constraint function, and why it is not squared
//!
//! Every constraint is a scalar `g(q) = 0` over the body positions `q`.
//! The only kind today is the rigid distance constraint:
//!
//! ```text
//! g(q) = |q_j - q_i| - L,     with gradient  ∂g/∂q_j = -∂g/∂q_i = d̂
//! ```
//!
//! where `d = q_j - q_i` and `d̂ = d/|d|`.
//!
//! The obvious alternative is the squared form `|d|² - L²`, which is a
//! polynomial and never divides by `|d|`. **It was tried first and it
//! does not work here.** Its gradient is `2d`, of magnitude `2L`, and its
//! value is in units of length *squared* — so in the DAE's iteration
//! matrix the constraint rows are scaled by `L` relative to the
//! differential rows. For `L = 1` that is invisible; for `L = 1.3` the
//! index-2 corrector stops converging and the step size collapses to
//! `1e-17`. The observed failure pattern was erratic in `L` and almost
//! nothing else, which is the signature of a conditioning problem rather
//! than a modelling one.
//!
//! The unsquared form has a **unit gradient** at every configuration and
//! a value in the same units as the coordinates, so every block of the
//! matrix carries the same scale. The division by `|d|` is safe because a
//! rod of length `L > 0` cannot reach `|d| = 0` without the constraint
//! already being violated by `L`; [`ConstraintSet::add_distance`] refuses
//! `L ≤ 0`, and the residual reports a recoverable error if it ever
//! happens anyway.
//!
//! `G` is very sparse — two blocks per row — which is why this module
//! applies it as a loop over constraints rather than materializing a
//! matrix.
//!
//! # Anchors
//!
//! A constraint whose partner has `inverse_mass == 0` (a wall, or any
//! body the user pinned) is a **fixed anchor**: the multipliers are not
//! allowed to push it. That is how a pendulum is built — anchor plus one
//! bob plus one rod.
//!
//! # Scope
//!
//! Constraints act on body **positions** only. A constraint that ought to
//! also grip a body's *orientation* (a hinge, a universal joint) is not
//! expressible here, and [`ConstraintSet::gate`] refuses the run with a
//! message saying so rather than silently integrating something else.
//! This mirrors the SPRK separability gate in [`crate::integrate`].

use crate::linalg::Vec3;
use crate::system::PhysicalObjectSystem;

/// A rigid distance constraint: `|q_j - q_i|` is held at `length`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DistanceConstraint {
    pub i: usize,
    pub j: usize,
    pub length: f64,
}

/// Every constraint acting on a system, in multiplier order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ConstraintSet {
    pub distances: Vec<DistanceConstraint>,
}

impl ConstraintSet {
    /// Number of scalar constraints — the number of Lagrange multipliers
    /// `λ`, and equally the number of GGL multipliers `μ`.
    pub fn len(&self) -> usize {
        self.distances.len()
    }

    pub fn is_empty(&self) -> bool {
        self.distances.is_empty()
    }

    /// Adds a rod between `i` and `j`. `length` defaults to the distance
    /// they are already at, which is almost always what the user means
    /// (`CONSTRAIN a b` freezes the current separation).
    ///
    /// Rejects the degenerate cases explicitly rather than producing a
    /// singular Jacobian three screens later.
    pub fn add_distance(
        &mut self,
        system: &PhysicalObjectSystem,
        i: usize,
        j: usize,
        length: Option<f64>,
    ) -> Result<usize, String> {
        let n = system.objects.len();
        if i >= n || j >= n {
            return Err(format!(
                "CONSTRAIN needs two existing objects (obj0..obj{}); got obj{i} and obj{j}",
                n.saturating_sub(1)
            ));
        }
        if i == j {
            return Err("CONSTRAIN needs two DIFFERENT objects".to_string());
        }
        if self
            .distances
            .iter()
            .any(|c| (c.i == i && c.j == j) || (c.i == j && c.j == i))
        {
            return Err(format!(
                "obj{i} and obj{j} are already constrained — DEL the constraint first \
                 (CONSTRAINTS lists them)"
            ));
        }
        let here = (system.objects[j].get_position() - system.objects[i].get_position()).norm();
        let length = length.unwrap_or(here);
        if !(length.is_finite() && length > 0.0) {
            return Err(format!(
                "CONSTRAIN length must be positive and finite (got {length})"
            ));
        }
        if system.objects[i].get_inverse_mass() == 0.0
            && system.objects[j].get_inverse_mass() == 0.0
        {
            return Err(format!(
                "obj{i} and obj{j} both have inverse_mass = 0 — a rod between two anchors \
                 constrains nothing and makes the DAE singular"
            ));
        }
        self.distances.push(DistanceConstraint { i, j, length });
        Ok(self.distances.len() - 1)
    }

    /// Refuses a constrained run whose physics this module cannot
    /// express, naming the offending feature — the same contract the SPRK
    /// gate follows.
    pub fn gate(&self, system: &PhysicalObjectSystem) -> Result<(), String> {
        for (k, o) in system.objects.iter().enumerate() {
            if o.get_angular_momentum() != Vec3::zeros()
                && o.get_inverse_inertia_tensor() != crate::linalg::Mat3::zeros()
            {
                return Err(format!(
                    "constrained (IDA) integration is translational only: obj{k} has spinning \
                     rigid-body state (nonzero angular momentum and invertible inertia). \
                     Constraints act on positions, not orientations; zero the spin, or drop the \
                     constraint and use METHOD ADAMS or BDF"
                ));
            }
        }
        for (k, tq) in system.external_torques.iter().enumerate() {
            if *tq != Vec3::zeros() {
                return Err(format!(
                    "constrained (IDA) integration cannot apply external torques (obj{k}): \
                     constraints act on positions only"
                ));
            }
        }
        Ok(())
    }

    /// `g(q) = |d| - L` for every constraint.
    pub fn residual(&self, q: &[f64], out: &mut [f64]) {
        for (k, c) in self.distances.iter().enumerate() {
            let d = read3(q, 3 * c.j) - read3(q, 3 * c.i);
            out[k] = d.norm() - c.length;
        }
    }

    /// `(G v)_k = d̂·(v_j - v_i)` — the time derivative `ġ`, which the GGL
    /// formulation drives to zero alongside `g` itself. This is the
    /// component of the relative velocity ALONG the rod: zero means the
    /// rod is neither stretching nor shortening.
    pub fn velocity_residual(&self, q: &[f64], v: &[f64], out: &mut [f64]) {
        for (k, c) in self.distances.iter().enumerate() {
            let d = read3(q, 3 * c.j) - read3(q, 3 * c.i);
            let dv = read3(v, 3 * c.j) - read3(v, 3 * c.i);
            out[k] = unit(d).dot(dv);
        }
    }

    /// Accumulates `Gᵀ m` into `out` (a `3N` block), skipping anchors.
    ///
    /// This is the only place the multipliers touch the bodies, and the
    /// anchor rule lives here: a body with `inverse_mass == 0` receives
    /// no multiplier force, so a rod to it is a pin joint to the world.
    pub fn add_jacobian_transpose(
        &self,
        anchors: &[bool],
        q: &[f64],
        m: &[f64],
        out: &mut [f64],
    ) {
        for (k, c) in self.distances.iter().enumerate() {
            let dhat = unit(read3(q, 3 * c.j) - read3(q, 3 * c.i));
            let w = m[k];
            if !anchors[c.i] {
                accumulate3(out, 3 * c.i, -w * dhat);
            }
            if !anchors[c.j] {
                accumulate3(out, 3 * c.j, w * dhat);
            }
        }
    }

    /// `anchors[k]` — body `k` has `inverse_mass == 0` and is therefore
    /// immovable. Multipliers are not allowed to push it, which is what
    /// turns a rod to it into a pin joint to the world.
    pub fn anchor_flags(system: &PhysicalObjectSystem) -> Vec<bool> {
        system.objects.iter().map(|o| o.get_inverse_mass() == 0.0).collect()
    }

    /// The worst `|g|` and `|ġ|` over the set, in the system's current
    /// configuration. This is the number that tells you whether the
    /// constraint is actually being *held* — see
    /// [`crate::integrate::RunReport::constraint_drift`].
    pub fn drift(&self, system: &PhysicalObjectSystem) -> (f64, f64) {
        let (mut g, mut gdot) = (0.0f64, 0.0f64);
        for c in &self.distances {
            let d = system.objects[c.j].get_position() - system.objects[c.i].get_position();
            let dv = system.objects[c.j].get_velocity() - system.objects[c.i].get_velocity();
            g = g.max((d.norm() - c.length).abs());
            gdot = gdot.max(unit(d).dot(dv).abs());
        }
        (g, gdot)
    }
}

/// `d/|d|`, or the zero vector when `d` is exactly zero. A rod can only
/// reach `|d| = 0` if it has already collapsed through its own length, so
/// this is a guard against producing a NaN, not a supported state.
fn unit(d: Vec3) -> Vec3 {
    let n = d.norm();
    if n == 0.0 {
        Vec3::zeros()
    } else {
        d * (1.0 / n)
    }
}

fn read3(d: &[f64], at: usize) -> Vec3 {
    Vec3::new(d[at], d[at + 1], d[at + 2])
}

fn accumulate3(d: &mut [f64], at: usize, v: Vec3) {
    d[at] += v.x;
    d[at + 1] += v.y;
    d[at + 2] += v.z;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physical_object::physical_object;

    fn two_bodies(sep: f64) -> PhysicalObjectSystem {
        let a = physical_object::new_point(0, 1.0, Vec3::zeros(), Vec3::zeros());
        let b = physical_object::new_point(1, 1.0, Vec3::new(sep, 0.0, 0.0), Vec3::zeros());
        PhysicalObjectSystem::new(vec![a, b], 0.0)
    }

    #[test]
    fn a_bare_constrain_freezes_the_current_separation() {
        let sys = two_bodies(2.5);
        let mut cs = ConstraintSet::default();
        cs.add_distance(&sys, 0, 1, None).unwrap();
        assert_eq!(cs.distances[0].length, 2.5);
        // g is exactly zero in the configuration it was measured from
        let q = vec![0.0, 0.0, 0.0, 2.5, 0.0, 0.0];
        let mut g = vec![0.0; 1];
        cs.residual(&q, &mut g);
        assert_eq!(g[0], 0.0);
    }

    /// The analytic gradient must match a central difference of `g`
    /// itself — the check that keeps `Gᵀ` honest under any change of
    /// constraint form.
    #[test]
    fn jacobian_transpose_matches_a_central_difference() {
        let sys = two_bodies(2.0);
        let mut cs = ConstraintSet::default();
        cs.add_distance(&sys, 0, 1, Some(2.0)).unwrap();
        let q = vec![0.1, -0.2, 0.3, 2.0, 0.4, -0.1];

        let mut analytic = vec![0.0; 6];
        let anchors = ConstraintSet::anchor_flags(&sys);
        cs.add_jacobian_transpose(&anchors, &q, &[1.0], &mut analytic);

        let h = 1e-6;
        for k in 0..6 {
            let (mut qp, mut qm) = (q.clone(), q.clone());
            qp[k] += h;
            qm[k] -= h;
            let (mut gp, mut gm) = (vec![0.0], vec![0.0]);
            cs.residual(&qp, &mut gp);
            cs.residual(&qm, &mut gm);
            let fd = (gp[0] - gm[0]) / (2.0 * h);
            assert!(
                (fd - analytic[k]).abs() < 1e-6,
                "component {k}: finite difference {fd} vs analytic {}",
                analytic[k]
            );
        }
    }

    /// An anchor absorbs the multiplier instead of being pushed by it.
    #[test]
    fn an_anchor_receives_no_multiplier_force() {
        let mut sys = two_bodies(1.0);
        sys.objects[0].set_inverse_mass(0.0);
        let mut cs = ConstraintSet::default();
        cs.add_distance(&sys, 0, 1, Some(1.0)).unwrap();
        let q = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let mut out = vec![0.0; 6];
        let anchors = ConstraintSet::anchor_flags(&sys);
        cs.add_jacobian_transpose(&anchors, &q, &[1.0], &mut out);
        assert_eq!(&out[0..3], &[0.0, 0.0, 0.0], "anchor must not be pushed");
        assert_eq!(&out[3..6], &[1.0, 0.0, 0.0], "free body carries the force");
    }

    #[test]
    fn degenerate_constraints_are_refused_by_name() {
        let mut sys = two_bodies(1.0);
        let mut cs = ConstraintSet::default();
        assert!(cs.add_distance(&sys, 0, 0, None).unwrap_err().contains("DIFFERENT"));
        assert!(cs.add_distance(&sys, 0, 9, None).unwrap_err().contains("existing objects"));
        assert!(cs.add_distance(&sys, 0, 1, Some(-1.0)).unwrap_err().contains("positive"));
        cs.add_distance(&sys, 0, 1, None).unwrap();
        assert!(cs.add_distance(&sys, 1, 0, None).unwrap_err().contains("already constrained"));

        sys.objects[0].set_inverse_mass(0.0);
        sys.objects[1].set_inverse_mass(0.0);
        let mut cs2 = ConstraintSet::default();
        assert!(cs2.add_distance(&sys, 0, 1, None).unwrap_err().contains("both have inverse_mass"));
    }
}
