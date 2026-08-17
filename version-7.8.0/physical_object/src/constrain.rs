//! Rigid holonomic constraints — rods, ball joints, hinges and universal
//! joints — and the algebra the DAE solvers need.
//!
//! A constraint is a geometric relation the motion must satisfy *exactly*
//! rather than a stiff spring that only approximately enforces it.
//! Enforcing it exactly turns the equations of motion from an ODE into a
//! **differential-algebraic equation** (DAE), which is what
//! `ida_rs`/`idas_rs` are for.
//!
//! # The four joints
//!
//! | joint | rows | holds | freedoms left |
//! |---|---|---|---|
//! | [`Joint::Distance`] | 1 | two bodies a fixed distance apart | 5 |
//! | [`Joint::Ball`] | 3 | a shared point | 3 (any rotation about it) |
//! | [`Joint::Universal`] | 4 | a shared point, two shafts kept square | 2 |
//! | [`Joint::Hinge`] | 5 | a shared point and a shared axis | 1 (the swing) |
//!
//! # Everything is a velocity Jacobian
//!
//! Each scalar constraint is a function `g` of the configuration that
//! must stay zero. What the solver needs is the row of the **velocity
//! Jacobian** `J` with
//!
//! ```text
//! ġ = J · u,        u = [v₀ ω₀ v₁ ω₁ …]
//! ```
//!
//! — how fast the constraint is being violated, given every body's
//! linear and angular velocity. That one abstraction covers all four
//! joints: a rod contributes only `J_v` blocks, an axis alignment only
//! `J_ω` blocks, a ball joint both. And the constraint *force* is `Jᵀλ`,
//! read out of the same rows: `J_vᵀλ` is a force, `J_ωᵀλ` a torque.
//!
//! ## The blocks, derived
//!
//! Writing `r = R a` for a body-frame attachment point carried into the
//! world, and `e_k` for the k-th axis:
//!
//! - **Distance**, `g = |d| - L` with `d = q_j - q_i`:
//!   `J_v,i = -d̂`, `J_v,j = +d̂`, no angular part.
//! - **Ball**, `g = (q_i + r_i) - (q_j + r_j)`, three rows:
//!   `ġ = v_i + ω_i×r_i - v_j - ω_j×r_j`. Since `e_k·(ω×r) = ω·(r×e_k)`,
//!   row `k` of the angular block is `(r×e_k)ᵀ` — the matrix `-[r]ₓ`.
//! - **Axis alignment**, `g = a·b` for two world unit vectors carried by
//!   the two bodies: `ġ = ω_i·(a×b) + ω_j·(b×a)`, so `J_ω,i = a×b` and
//!   `J_ω,j = -(a×b)`, no linear part.
//!
//! A hinge is a ball joint plus **two** alignment rows (its axis against
//! two vectors spanning the plane perpendicular to it in the other
//! body); a universal joint is a ball joint plus **one** (its two shafts
//! against each other).
//!
//! # Why nothing here is squared
//!
//! The distance constraint is `|d| - L`, not `|d|² - L²`. The squared
//! form is a polynomial and never divides — and it does not work. Its
//! gradient is `2d`, of magnitude `2L`, and its value is in units of
//! length *squared*, so in the DAE's iteration matrix the constraint rows
//! are scaled by `L` against the differential rows. At `L = 1` that is
//! invisible; at `L = 1.3` the index-2 corrector stops converging and the
//! step collapses to `1e-17`. Every gradient here is deliberately
//! **O(1)**: unit vectors, identity blocks, and skew matrices of the
//! attachment arms. Keep it that way.
//!
//! The one scale that is *not* dimensionless is the ball rows, which are
//! in length units while the alignment rows are pure numbers. For a joint
//! whose arms are of order the body size that is a ratio near 1. A
//! mechanism built at wildly mixed scales would want row scaling; none
//! here does.
//!
//! # Anchors
//!
//! A body with `inverse_mass == 0` is a **translational anchor**: the
//! multipliers may not push it. A body whose inverse inertia tensor is
//! zero is a **rotational anchor**: they may not twist it. A wall is
//! both — which is what makes a hinge to a wall a door.

use crate::linalg::{Mat3, Quat, Vec3};
use crate::system::PhysicalObjectSystem;

/// One rigid joint between two bodies.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Joint {
    /// `|q_j - q_i| = length`. Leaves five degrees of freedom.
    Distance { i: usize, j: usize, length: f64 },
    /// The two bodies share a point, held in each one's own frame.
    Ball { i: usize, j: usize, a_i: Vec3, a_j: Vec3 },
    /// A ball joint plus the hinge axis `h_i` (body `i`) held parallel to
    /// the axis it started on in body `j`, enforced as `h_i ⟂ p_j` and
    /// `h_i ⟂ q_j` for two vectors spanning the perpendicular plane.
    Hinge { i: usize, j: usize, a_i: Vec3, a_j: Vec3, h_i: Vec3, p_j: Vec3, q_j: Vec3 },
    /// A ball joint plus the two cross-shafts held square — a Cardan joint.
    Universal { i: usize, j: usize, a_i: Vec3, a_j: Vec3, u_i: Vec3, u_j: Vec3 },
}

impl Joint {
    /// Number of scalar constraints — equivalently, degrees of freedom
    /// removed from the pair's six.
    pub fn rows(&self) -> usize {
        match self {
            Joint::Distance { .. } => 1,
            Joint::Ball { .. } => 3,
            Joint::Universal { .. } => 4,
            Joint::Hinge { .. } => 5,
        }
    }

    pub fn bodies(&self) -> (usize, usize) {
        match *self {
            Joint::Distance { i, j, .. }
            | Joint::Ball { i, j, .. }
            | Joint::Hinge { i, j, .. }
            | Joint::Universal { i, j, .. } => (i, j),
        }
    }

    pub fn kind(&self) -> &'static str {
        match self {
            Joint::Distance { .. } => "rod",
            Joint::Ball { .. } => "ball",
            Joint::Hinge { .. } => "hinge",
            Joint::Universal { .. } => "universal",
        }
    }

    /// Does this joint constrain orientation?
    pub fn is_rotational(&self) -> bool {
        !matches!(self, Joint::Distance { .. })
    }
}

/// One `(body, J_v, J_ω)` block of one Jacobian row.
#[derive(Clone, Copy, Debug)]
pub struct JacBlock {
    pub body: usize,
    pub jv: Vec3,
    pub jw: Vec3,
}

/// The pose of one body, as the Jacobian needs it.
#[derive(Clone, Copy, Debug)]
pub struct Pose {
    pub position: Vec3,
    pub orientation: Quat,
}

/// Which bodies the multipliers may not move.
#[derive(Clone, Debug, Default)]
pub struct Anchors {
    /// `inverse_mass == 0` — cannot be pushed.
    pub translation_fixed: Vec<bool>,
    /// zero inverse inertia — cannot be twisted (a point mass, or a wall).
    pub rotation_fixed: Vec<bool>,
}

impl Anchors {
    pub fn of(system: &PhysicalObjectSystem) -> Self {
        Self {
            translation_fixed: system.objects.iter().map(|o| o.get_inverse_mass() == 0.0).collect(),
            rotation_fixed: system
                .objects
                .iter()
                .map(|o| o.get_inverse_inertia_tensor() == Mat3::zeros())
                .collect(),
        }
    }
}

/// Every joint acting on a system, in multiplier order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ConstraintSet {
    pub joints: Vec<Joint>,
}

impl ConstraintSet {
    /// Total scalar constraints — the number of `λ`, and equally of `μ`.
    pub fn len(&self) -> usize {
        self.joints.iter().map(|j| j.rows()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.joints.is_empty()
    }

    /// Does any joint constrain orientation? Only the DAE path carries
    /// orientation in its state, so this is what the other solvers gate on.
    pub fn has_rotational(&self) -> bool {
        self.joints.iter().any(|j| j.is_rotational())
    }

    /* --- construction --------------------------------------------- */

    fn check_pair(
        &self,
        system: &PhysicalObjectSystem,
        i: usize,
        j: usize,
        what: &str,
    ) -> Result<(), String> {
        let n = system.objects.len();
        if i >= n || j >= n {
            return Err(format!(
                "{what} needs two existing objects (obj0..obj{}); got obj{i} and obj{j}",
                n.saturating_sub(1)
            ));
        }
        if i == j {
            return Err(format!("{what} needs two DIFFERENT objects"));
        }
        if self.joints.iter().any(|c| {
            let (a, b) = c.bodies();
            (a == i && b == j) || (a == j && b == i)
        }) {
            return Err(format!(
                "obj{i} and obj{j} are already joined — CONSTRAIN OFF drops every joint \
                 (CONSTRAINTS lists them)"
            ));
        }
        if system.objects[i].get_inverse_mass() == 0.0
            && system.objects[j].get_inverse_mass() == 0.0
        {
            return Err(format!(
                "obj{i} and obj{j} both have inverse_mass = 0 — a joint between two anchors \
                 constrains nothing and makes the DAE singular"
            ));
        }
        Ok(())
    }

    /// A rigid rod. `length` defaults to the separation the bodies
    /// already have, which is always consistent.
    pub fn add_distance(
        &mut self,
        system: &PhysicalObjectSystem,
        i: usize,
        j: usize,
        length: Option<f64>,
    ) -> Result<usize, String> {
        self.check_pair(system, i, j, "CONSTRAIN")?;
        let here = (system.objects[j].get_position() - system.objects[i].get_position()).norm();
        let length = length.unwrap_or(here);
        if !(length.is_finite() && length > 0.0) {
            return Err(format!("CONSTRAIN length must be positive and finite (got {length})"));
        }
        self.joints.push(Joint::Distance { i, j, length });
        Ok(self.joints.len() - 1)
    }

    /// The pivot every joint below is built around: the midpoint of the
    /// two bodies' current positions, carried into each body's own frame.
    ///
    /// Freezing the current configuration is the principle a bare
    /// `CONSTRAIN` already follows — it guarantees the joint is satisfied
    /// the instant it is made, so there is never an inconsistent initial
    /// condition to repair. Place the bodies where you want the pivot.
    fn pivot_arms(system: &PhysicalObjectSystem, i: usize, j: usize) -> (Vec3, Vec3) {
        let pi = system.objects[i].get_position();
        let pj = system.objects[j].get_position();
        let pivot = 0.5 * (pi + pj);
        let ri = system.objects[i].get_orientation().normalize().inverse();
        let rj = system.objects[j].get_orientation().normalize().inverse();
        (ri.rotate(pivot - pi), rj.rotate(pivot - pj))
    }

    /// A ball (spherical) joint at the midpoint of the two bodies.
    pub fn add_ball(
        &mut self,
        system: &PhysicalObjectSystem,
        i: usize,
        j: usize,
    ) -> Result<usize, String> {
        self.check_pair(system, i, j, "BALL")?;
        let (a_i, a_j) = Self::pivot_arms(system, i, j);
        self.joints.push(Joint::Ball { i, j, a_i, a_j });
        Ok(self.joints.len() - 1)
    }

    /// A hinge at the midpoint, turning about the world-frame `axis` as
    /// it stands now.
    pub fn add_hinge(
        &mut self,
        system: &PhysicalObjectSystem,
        i: usize,
        j: usize,
        axis: Vec3,
    ) -> Result<usize, String> {
        self.check_pair(system, i, j, "HINGE")?;
        if !(axis.norm() > 0.0 && axis.norm().is_finite()) {
            return Err(
                "HINGE needs a non-zero, finite axis, e.g. `hinge a b [0, 0, 1]`".to_string()
            );
        }
        let h = axis.normalize();
        let (p, q) = perpendicular_basis(h);
        let (a_i, a_j) = Self::pivot_arms(system, i, j);
        let ri = system.objects[i].get_orientation().normalize().inverse();
        let rj = system.objects[j].get_orientation().normalize().inverse();
        self.joints.push(Joint::Hinge {
            i,
            j,
            a_i,
            a_j,
            h_i: ri.rotate(h),
            p_j: rj.rotate(p),
            q_j: rj.rotate(q),
        });
        Ok(self.joints.len() - 1)
    }

    /// A universal (Cardan) joint at the midpoint. `axis_i` is carried by
    /// body `i`, `axis_j` by body `j`, and the joint holds them square.
    pub fn add_universal(
        &mut self,
        system: &PhysicalObjectSystem,
        i: usize,
        j: usize,
        axis_i: Vec3,
        axis_j: Vec3,
    ) -> Result<usize, String> {
        self.check_pair(system, i, j, "UNIVERSAL")?;
        for (name, a) in [("first", axis_i), ("second", axis_j)] {
            if !(a.norm() > 0.0 && a.norm().is_finite()) {
                return Err(format!(
                    "UNIVERSAL needs two non-zero, finite axes; the {name} one is {a:?}"
                ));
            }
        }
        let u = axis_i.normalize();
        let w = axis_j.normalize();
        /* The constraint IS u·w = 0, so it must hold at creation or the
         * joint starts violated. Rather than silently projecting one axis
         * onto the other — which would build a different mechanism than
         * the one asked for — say so, with the angle actually found. */
        let dot = u.dot(w);
        if dot.abs() > 1e-9 {
            return Err(format!(
                "UNIVERSAL holds its two axes perpendicular, but they start {:.4}° apart \
                 (dot product {dot}). Pick axes 90° apart, e.g. \
                 `universal a b [1, 0, 0] [0, 1, 0]`",
                dot.clamp(-1.0, 1.0).acos().to_degrees()
            ));
        }
        let (a_i, a_j) = Self::pivot_arms(system, i, j);
        let ri = system.objects[i].get_orientation().normalize().inverse();
        let rj = system.objects[j].get_orientation().normalize().inverse();
        self.joints.push(Joint::Universal {
            i,
            j,
            a_i,
            a_j,
            u_i: ri.rotate(u),
            u_j: rj.rotate(w),
        });
        Ok(self.joints.len() - 1)
    }

    /* --- the algebra ---------------------------------------------- */

    /// `g` for every scalar constraint, given every body's pose.
    pub fn residual(&self, pose: &[Pose], out: &mut [f64]) {
        let mut r = 0;
        for joint in &self.joints {
            match *joint {
                Joint::Distance { i, j, length } => {
                    out[r] = (pose[j].position - pose[i].position).norm() - length;
                    r += 1;
                }
                Joint::Ball { i, j, a_i, a_j } => {
                    write3(out, r, ball_gap(pose, i, j, a_i, a_j));
                    r += 3;
                }
                Joint::Hinge { i, j, a_i, a_j, h_i, p_j, q_j } => {
                    write3(out, r, ball_gap(pose, i, j, a_i, a_j));
                    let h = rot(pose, i, h_i);
                    out[r + 3] = h.dot(rot(pose, j, p_j));
                    out[r + 4] = h.dot(rot(pose, j, q_j));
                    r += 5;
                }
                Joint::Universal { i, j, a_i, a_j, u_i, u_j } => {
                    write3(out, r, ball_gap(pose, i, j, a_i, a_j));
                    out[r + 3] = rot(pose, i, u_i).dot(rot(pose, j, u_j));
                    r += 4;
                }
            }
        }
    }

    /// Calls `emit(row, block)` for every non-zero block of the velocity
    /// Jacobian at this configuration. Everything else is written in
    /// terms of this one walk.
    pub fn for_each_block(&self, pose: &[Pose], mut emit: impl FnMut(usize, JacBlock)) {
        let mut r = 0;
        for joint in &self.joints {
            match *joint {
                Joint::Distance { i, j, .. } => {
                    let dhat = unit(pose[j].position - pose[i].position);
                    emit(r, JacBlock { body: i, jv: -dhat, jw: Vec3::zeros() });
                    emit(r, JacBlock { body: j, jv: dhat, jw: Vec3::zeros() });
                    r += 1;
                }
                Joint::Ball { i, j, a_i, a_j } => {
                    ball_blocks(pose, i, j, a_i, a_j, r, &mut emit);
                    r += 3;
                }
                Joint::Hinge { i, j, a_i, a_j, h_i, p_j, q_j } => {
                    ball_blocks(pose, i, j, a_i, a_j, r, &mut emit);
                    let h = rot(pose, i, h_i);
                    for (k, body_axis) in [p_j, q_j].into_iter().enumerate() {
                        let c = h.cross(rot(pose, j, body_axis));
                        emit(r + 3 + k, JacBlock { body: i, jv: Vec3::zeros(), jw: c });
                        emit(r + 3 + k, JacBlock { body: j, jv: Vec3::zeros(), jw: -c });
                    }
                    r += 5;
                }
                Joint::Universal { i, j, a_i, a_j, u_i, u_j } => {
                    ball_blocks(pose, i, j, a_i, a_j, r, &mut emit);
                    let c = rot(pose, i, u_i).cross(rot(pose, j, u_j));
                    emit(r + 3, JacBlock { body: i, jv: Vec3::zeros(), jw: c });
                    emit(r + 3, JacBlock { body: j, jv: Vec3::zeros(), jw: -c });
                    r += 4;
                }
            }
        }
    }

    /// `ġ = J·u` for every constraint.
    pub fn velocity_residual(&self, pose: &[Pose], v: &[Vec3], w: &[Vec3], out: &mut [f64]) {
        for o in out.iter_mut() {
            *o = 0.0;
        }
        self.for_each_block(pose, |row, b| {
            out[row] += b.jv.dot(v[b.body]) + b.jw.dot(w[b.body]);
        });
    }

    /// Accumulates `Jᵀm` — a force and a torque per body.
    ///
    /// The anchor rules live here and nowhere else: a translational
    /// anchor takes no force, a rotational anchor no torque. That is what
    /// turns a hinge to a wall into a door.
    pub fn add_jacobian_transpose(
        &self,
        pose: &[Pose],
        anchors: &Anchors,
        m: &[f64],
        force: &mut [Vec3],
        torque: &mut [Vec3],
    ) {
        self.for_each_block(pose, |row, b| {
            let lam = m[row];
            if !anchors.translation_fixed[b.body] {
                force[b.body] += lam * b.jv;
            }
            if !anchors.rotation_fixed[b.body] {
                torque[b.body] += lam * b.jw;
            }
        });
    }

    /// The worst `|g|` and `|ġ|` over the whole set in the system's
    /// current configuration — how far the joints have actually strayed.
    pub fn drift(&self, system: &PhysicalObjectSystem) -> (f64, f64) {
        if self.is_empty() {
            return (0.0, 0.0);
        }
        let pose = Self::poses(system);
        let v: Vec<Vec3> = system.objects.iter().map(|o| o.get_velocity()).collect();
        let w: Vec<Vec3> = system.objects.iter().map(angular_velocity).collect();
        let m = self.len();
        let (mut g, mut gd) = (vec![0.0; m], vec![0.0; m]);
        self.residual(&pose, &mut g);
        self.velocity_residual(&pose, &v, &w, &mut gd);
        (
            g.iter().fold(0.0f64, |a, x| a.max(x.abs())),
            gd.iter().fold(0.0f64, |a, x| a.max(x.abs())),
        )
    }

    pub fn poses(system: &PhysicalObjectSystem) -> Vec<Pose> {
        system
            .objects
            .iter()
            .map(|o| Pose { position: o.get_position(), orientation: o.get_orientation() })
            .collect()
    }
}

/// World-frame angular velocity `ω = R I⁻¹ Rᵀ L` — the same expression
/// `integrate::rhs_full` uses, in the same order.
pub fn angular_velocity(o: &crate::physical_object::physical_object) -> Vec3 {
    let r = o.get_orientation().normalize().to_rotation_matrix();
    r * o.get_inverse_inertia_tensor() * r.transpose() * o.get_angular_momentum()
}

fn rot(pose: &[Pose], k: usize, v: Vec3) -> Vec3 {
    pose[k].orientation.normalize().rotate(v)
}

fn write3(out: &mut [f64], at: usize, v: Vec3) {
    out[at] = v.x;
    out[at + 1] = v.y;
    out[at + 2] = v.z;
}

fn ball_gap(pose: &[Pose], i: usize, j: usize, a_i: Vec3, a_j: Vec3) -> Vec3 {
    (pose[i].position + rot(pose, i, a_i)) - (pose[j].position + rot(pose, j, a_j))
}

fn ball_blocks(
    pose: &[Pose],
    i: usize,
    j: usize,
    a_i: Vec3,
    a_j: Vec3,
    row0: usize,
    emit: &mut impl FnMut(usize, JacBlock),
) {
    let ri = rot(pose, i, a_i);
    let rj = rot(pose, j, a_j);
    /* Row k of the angular block is (r × e_k)ᵀ — the matrix -[r]ₓ —
     * because e_k·(ω×r) = ω·(r×e_k). */
    const AXES: [Vec3; 3] = [
        Vec3 { x: 1.0, y: 0.0, z: 0.0 },
        Vec3 { x: 0.0, y: 1.0, z: 0.0 },
        Vec3 { x: 0.0, y: 0.0, z: 1.0 },
    ];
    for (k, e) in AXES.into_iter().enumerate() {
        emit(row0 + k, JacBlock { body: i, jv: e, jw: ri.cross(e) });
        emit(row0 + k, JacBlock { body: j, jv: -e, jw: -rj.cross(e) });
    }
}

/// Two unit vectors completing `h` to an orthonormal basis. Seeds from
/// the world axis least aligned with `h`, so the cross product is never
/// near-degenerate.
fn perpendicular_basis(h: Vec3) -> (Vec3, Vec3) {
    let seed = if h.x.abs() <= h.y.abs() && h.x.abs() <= h.z.abs() {
        Vec3::new(1.0, 0.0, 0.0)
    } else if h.y.abs() <= h.z.abs() {
        Vec3::new(0.0, 1.0, 0.0)
    } else {
        Vec3::new(0.0, 0.0, 1.0)
    };
    let p = h.cross(seed).normalize();
    (p, h.cross(p))
}

fn unit(d: Vec3) -> Vec3 {
    let n = d.norm();
    if n == 0.0 {
        Vec3::zeros()
    } else {
        d * (1.0 / n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::Boundary;
    use crate::physical_object::physical_object;

    fn two_boxes(sep: f64) -> PhysicalObjectSystem {
        let cuboid = || Boundary::Cuboid { half_extents: [0.5, 0.4, 0.3] };
        let a = physical_object::new_from_shape(
            0, 1.0, 0.0, Vec3::zeros(), Vec3::zeros(), Vec3::zeros(), cuboid(),
        );
        let b = physical_object::new_from_shape(
            1, 1.0, 0.0, Vec3::new(sep, 0.0, 0.0), Vec3::zeros(), Vec3::zeros(), cuboid(),
        );
        PhysicalObjectSystem::new(vec![a, b], 0.0)
    }

    fn with<F: Fn(&mut ConstraintSet, &PhysicalObjectSystem) -> Result<usize, String>>(
        sys: &PhysicalObjectSystem,
        f: F,
    ) -> ConstraintSet {
        let mut cs = ConstraintSet::default();
        f(&mut cs, sys).unwrap();
        cs
    }

    /// Every joint is built from the configuration the bodies are already
    /// in, so it is satisfied the instant it is made — there is never an
    /// inconsistent initial condition for the DAE to repair.
    #[test]
    fn every_joint_starts_satisfied() {
        let sys = two_boxes(2.0);
        let sets = [
            with(&sys, |c, s| c.add_distance(s, 0, 1, None)),
            with(&sys, |c, s| c.add_ball(s, 0, 1)),
            with(&sys, |c, s| c.add_hinge(s, 0, 1, Vec3::new(0.0, 0.0, 1.0))),
            with(&sys, |c, s| {
                c.add_universal(s, 0, 1, Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0))
            }),
        ];
        for cs in sets {
            let (g, gd) = cs.drift(&sys);
            assert!(g < 1e-15, "{} starts at |g| = {g:e}", cs.joints[0].kind());
            assert!(gd < 1e-15, "{} starts at |g_dot| = {gd:e}", cs.joints[0].kind());
        }
    }

    /// The row count IS the number of degrees of freedom removed from the
    /// pair's six.
    #[test]
    fn row_counts_are_the_freedoms_removed() {
        let sys = two_boxes(2.0);
        assert_eq!(with(&sys, |c, s| c.add_distance(s, 0, 1, None)).len(), 1);
        assert_eq!(with(&sys, |c, s| c.add_ball(s, 0, 1)).len(), 3);
        assert_eq!(
            with(&sys, |c, s| c.add_universal(
                s, 0, 1, Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0)
            ))
            .len(),
            4
        );
        assert_eq!(with(&sys, |c, s| c.add_hinge(s, 0, 1, Vec3::new(0.0, 0.0, 1.0))).len(), 5);
        assert!(!with(&sys, |c, s| c.add_distance(s, 0, 1, None)).has_rotational());
        assert!(with(&sys, |c, s| c.add_ball(s, 0, 1)).has_rotational());
    }

    /// The velocity Jacobian must be the honest derivative of `g`: move
    /// the configuration along a velocity for a tiny time and `g` must
    /// change at exactly the rate `J·u` predicts. A wrong sign, a
    /// transposed skew matrix or a mixed-up body index all fail here.
    #[test]
    fn the_velocity_jacobian_matches_a_finite_difference_of_g() {
        let sys = two_boxes(2.0);
        let mut cs = ConstraintSet::default();
        cs.add_hinge(&sys, 0, 1, Vec3::new(0.3, -0.5, 0.81)).unwrap();
        let m = cs.len();

        // an arbitrary, deliberately messy motion
        let v = [Vec3::new(0.3, -0.7, 0.2), Vec3::new(-0.1, 0.45, 0.9)];
        let w = [Vec3::new(0.8, 0.2, -0.4), Vec3::new(-0.25, 0.6, 0.15)];

        let pose0 = ConstraintSet::poses(&sys);
        let mut predicted = vec![0.0; m];
        cs.velocity_residual(&pose0, &v, &w, &mut predicted);

        let advance = |dt: f64| -> Vec<Pose> {
            pose0
                .iter()
                .enumerate()
                .map(|(k, p)| Pose {
                    position: p.position + dt * v[k],
                    // q(t+dt) ~ q + dt * 1/2 (0,w) x q, renormalised
                    orientation: (p.orientation + (Quat::pure(w[k]) * p.orientation) * (0.5 * dt))
                        .normalize(),
                })
                .collect()
        };
        let h = 1e-7;
        let (mut gp, mut gm) = (vec![0.0; m], vec![0.0; m]);
        cs.residual(&advance(h), &mut gp);
        cs.residual(&advance(-h), &mut gm);
        for k in 0..m {
            let fd = (gp[k] - gm[k]) / (2.0 * h);
            assert!(
                (fd - predicted[k]).abs() < 1e-6,
                "row {k}: finite difference {fd} vs Jacobian {}",
                predicted[k]
            );
        }
    }

    /// A hinge leaves exactly one rotational freedom: turning about its
    /// own axis violates nothing, turning about anything else does.
    #[test]
    fn a_hinge_admits_its_own_axis_and_refuses_the_others() {
        let sys = two_boxes(2.0);
        let axis = Vec3::new(0.0, 0.0, 1.0);
        let cs = with(&sys, |c, s| c.add_hinge(s, 0, 1, axis));
        let pose = ConstraintSet::poses(&sys);
        let mut g = vec![0.0; cs.len()];
        let still = [Vec3::zeros(); 2];

        cs.velocity_residual(&pose, &still, &[Vec3::zeros(), axis], &mut g);
        assert!(
            g[3].abs() < 1e-15 && g[4].abs() < 1e-15,
            "turning about the hinge axis is free: {g:?}"
        );

        cs.velocity_residual(&pose, &still, &[Vec3::zeros(), Vec3::new(1.0, 0.0, 0.0)], &mut g);
        assert!(
            g[3].abs() > 0.1 || g[4].abs() > 0.1,
            "tilting off the hinge axis must be resisted: {g:?}"
        );
    }

    /// A universal joint admits a turn about either shaft but resists the
    /// swing that would close the angle between them.
    #[test]
    fn a_universal_joint_admits_both_shafts() {
        let sys = two_boxes(2.0);
        let (u, w) = (Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
        let cs = with(&sys, |c, s| c.add_universal(s, 0, 1, u, w));
        let pose = ConstraintSet::poses(&sys);
        let mut g = vec![0.0; cs.len()];
        let still = [Vec3::zeros(); 2];

        cs.velocity_residual(&pose, &still, &[u, Vec3::zeros()], &mut g);
        assert!(g[3].abs() < 1e-15, "turning about its own shaft is free: {}", g[3]);
        cs.velocity_residual(&pose, &still, &[Vec3::zeros(), w], &mut g);
        assert!(g[3].abs() < 1e-15, "turning about its own shaft is free: {}", g[3]);
        cs.velocity_residual(&pose, &still, &[u.cross(w), Vec3::zeros()], &mut g);
        assert!(g[3].abs() > 0.5, "closing the shaft angle must be resisted: {}", g[3]);
    }

    #[test]
    fn anchors_take_no_force_and_no_torque() {
        let mut sys = two_boxes(2.0);
        sys.objects[0].set_inverse_mass(0.0);
        sys.objects[0].set_inertia_tensor(Mat3::zeros());
        let cs = with(&sys, |c, s| c.add_ball(s, 0, 1));
        let pose = ConstraintSet::poses(&sys);
        let anchors = Anchors::of(&sys);
        assert!(anchors.translation_fixed[0] && anchors.rotation_fixed[0]);

        let (mut f, mut t) = (vec![Vec3::zeros(); 2], vec![Vec3::zeros(); 2]);
        cs.add_jacobian_transpose(&pose, &anchors, &[1.0, 1.0, 1.0], &mut f, &mut t);
        assert_eq!(f[0], Vec3::zeros(), "anchor takes no force");
        assert_eq!(t[0], Vec3::zeros(), "anchor takes no torque");
        assert_ne!(f[1], Vec3::zeros(), "the free body carries it");
    }

    #[test]
    fn degenerate_joints_are_refused_by_name() {
        let mut sys = two_boxes(1.0);
        let mut cs = ConstraintSet::default();
        assert!(cs.add_ball(&sys, 0, 0).unwrap_err().contains("DIFFERENT"));
        assert!(cs
            .add_hinge(&sys, 0, 9, Vec3::new(0.0, 0.0, 1.0))
            .unwrap_err()
            .contains("existing"));
        assert!(cs.add_hinge(&sys, 0, 1, Vec3::zeros()).unwrap_err().contains("non-zero"));
        // shafts that are not square name the angle they actually found
        let e = cs
            .add_universal(&sys, 0, 1, Vec3::new(1.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 0.0))
            .unwrap_err();
        assert!(e.contains("perpendicular") && e.contains("45"), "{e}");

        cs.add_ball(&sys, 0, 1).unwrap();
        assert!(cs
            .add_hinge(&sys, 1, 0, Vec3::new(0.0, 0.0, 1.0))
            .unwrap_err()
            .contains("already joined"));

        sys.objects[0].set_inverse_mass(0.0);
        sys.objects[1].set_inverse_mass(0.0);
        let mut cs2 = ConstraintSet::default();
        assert!(cs2.add_ball(&sys, 0, 1).unwrap_err().contains("both have inverse_mass"));
    }
}
