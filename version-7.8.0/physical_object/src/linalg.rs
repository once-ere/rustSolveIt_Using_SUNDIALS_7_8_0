//! Minimal pure-std linear algebra: `Vec3`, `Mat3`, `Quat`.
//!
//! Replaces the `nalgebra` types used by the legacy `PointParticle` and
//! `RigidBody` (`Vector3<f64>`, `Matrix3<f64>`, `UnitQuaternion<f64>`)
//! and the hand-rolled `Matrix3x3` of `RigidBody3D`.

use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};

/// A 3-component column vector.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn zeros() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }

    pub fn from_array(a: [f64; 3]) -> Self {
        Self::new(a[0], a[1], a[2])
    }

    pub fn to_array(self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }

    pub fn dot(self, rhs: Vec3) -> f64 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    pub fn cross(self, rhs: Vec3) -> Vec3 {
        Vec3::new(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
    }

    pub fn norm_squared(self) -> f64 {
        self.dot(self)
    }

    pub fn norm(self) -> f64 {
        self.norm_squared().sqrt()
    }

    /// Unit vector in the same direction; the zero vector normalizes to
    /// the zero vector (matching the guard in the legacy `laplace_vector`).
    pub fn normalize(self) -> Vec3 {
        let n = self.norm();
        if n > 0.0 {
            self / n
        } else {
            Vec3::zeros()
        }
    }
}

impl Add for Vec3 {
    type Output = Vec3;
    fn add(self, rhs: Vec3) -> Vec3 {
        Vec3::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, rhs: Vec3) -> Vec3 {
        Vec3::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl Neg for Vec3 {
    type Output = Vec3;
    fn neg(self) -> Vec3 {
        Vec3::new(-self.x, -self.y, -self.z)
    }
}

impl Mul<f64> for Vec3 {
    type Output = Vec3;
    fn mul(self, s: f64) -> Vec3 {
        Vec3::new(self.x * s, self.y * s, self.z * s)
    }
}

impl Mul<Vec3> for f64 {
    type Output = Vec3;
    fn mul(self, v: Vec3) -> Vec3 {
        v * self
    }
}

impl Div<f64> for Vec3 {
    type Output = Vec3;
    fn div(self, s: f64) -> Vec3 {
        Vec3::new(self.x / s, self.y / s, self.z / s)
    }
}

impl AddAssign for Vec3 {
    fn add_assign(&mut self, rhs: Vec3) {
        *self = *self + rhs;
    }
}

impl SubAssign for Vec3 {
    fn sub_assign(&mut self, rhs: Vec3) {
        *self = *self - rhs;
    }
}

/// A 3x3 matrix, row-major.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mat3(pub [[f64; 3]; 3]);

impl Mat3 {
    pub fn zeros() -> Self {
        Mat3([[0.0; 3]; 3])
    }

    pub fn identity() -> Self {
        Mat3([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    }

    pub fn from_diagonal(d: Vec3) -> Self {
        Mat3([[d.x, 0.0, 0.0], [0.0, d.y, 0.0], [0.0, 0.0, d.z]])
    }

    pub fn transpose(self) -> Mat3 {
        let m = self.0;
        Mat3([
            [m[0][0], m[1][0], m[2][0]],
            [m[0][1], m[1][1], m[2][1]],
            [m[0][2], m[1][2], m[2][2]],
        ])
    }

    pub fn determinant(self) -> f64 {
        let m = self.0;
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    }

    /// General inverse via the adjugate; `None` when singular.
    pub fn try_inverse(self) -> Option<Mat3> {
        let det = self.determinant();
        if det == 0.0 || !det.is_finite() {
            return None;
        }
        let m = self.0;
        let inv_det = 1.0 / det;
        let c = |a: f64, b: f64, cc: f64, d: f64| (a * d - b * cc) * inv_det;
        Some(Mat3([
            [
                c(m[1][1], m[1][2], m[2][1], m[2][2]),
                c(m[0][2], m[0][1], m[2][2], m[2][1]),
                c(m[0][1], m[0][2], m[1][1], m[1][2]),
            ],
            [
                c(m[1][2], m[1][0], m[2][2], m[2][0]),
                c(m[0][0], m[0][2], m[2][0], m[2][2]),
                c(m[0][2], m[0][0], m[1][2], m[1][0]),
            ],
            [
                c(m[1][0], m[1][1], m[2][0], m[2][1]),
                c(m[0][1], m[0][0], m[2][1], m[2][0]),
                c(m[0][0], m[0][1], m[1][0], m[1][1]),
            ],
        ]))
    }

    /// Inverse of a diagonal matrix with the legacy `RigidBody3D`
    /// convention: a zero diagonal entry inverts to zero.
    pub fn inverse_diagonal(self) -> Mat3 {
        let m = self.0;
        let inv = |v: f64| if v == 0.0 { 0.0 } else { 1.0 / v };
        Mat3::from_diagonal(Vec3::new(inv(m[0][0]), inv(m[1][1]), inv(m[2][2])))
    }

    /// The cross-product (skew-symmetric) matrix `[v]×`, defined by
    /// `Mat3::skew(a) * b == a.cross(b)` for all `b`. Used to assemble
    /// the contact-solver effective-mass matrix
    /// `K = m⁻¹·1 − [r]× I_w⁻¹ [r]×`.
    pub fn skew(v: Vec3) -> Mat3 {
        Mat3([
            [0.0, -v.z, v.y],
            [v.z, 0.0, -v.x],
            [-v.y, v.x, 0.0],
        ])
    }

    /// The outer product `a bᵀ` (a rank-one matrix), defined by
    /// `Mat3::outer(a, b) * c == a * b.dot(c)` for all `c`.
    pub fn outer(a: Vec3, b: Vec3) -> Mat3 {
        Mat3([
            [a.x * b.x, a.x * b.y, a.x * b.z],
            [a.y * b.x, a.y * b.y, a.y * b.z],
            [a.z * b.x, a.z * b.y, a.z * b.z],
        ])
    }
}

impl Add for Mat3 {
    type Output = Mat3;
    fn add(self, rhs: Mat3) -> Mat3 {
        let mut out = self.0;
        for (row, r2) in out.iter_mut().zip(rhs.0.iter()) {
            for (cell, c2) in row.iter_mut().zip(r2.iter()) {
                *cell += *c2;
            }
        }
        Mat3(out)
    }
}

impl Sub for Mat3 {
    type Output = Mat3;
    fn sub(self, rhs: Mat3) -> Mat3 {
        let mut out = self.0;
        for (row, r2) in out.iter_mut().zip(rhs.0.iter()) {
            for (cell, c2) in row.iter_mut().zip(r2.iter()) {
                *cell -= *c2;
            }
        }
        Mat3(out)
    }
}

impl AddAssign for Mat3 {
    fn add_assign(&mut self, rhs: Mat3) {
        *self = *self + rhs;
    }
}

impl Mul<Vec3> for Mat3 {
    type Output = Vec3;
    fn mul(self, v: Vec3) -> Vec3 {
        let m = self.0;
        Vec3::new(
            m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
            m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
            m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z,
        )
    }
}

impl Mul<Mat3> for Mat3 {
    type Output = Mat3;
    fn mul(self, rhs: Mat3) -> Mat3 {
        let a = self.0;
        let b = rhs.0;
        let mut out = [[0.0; 3]; 3];
        for (i, row) in out.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
            }
        }
        Mat3(out)
    }
}

impl Mul<f64> for Mat3 {
    type Output = Mat3;
    fn mul(self, s: f64) -> Mat3 {
        let mut out = self.0;
        for row in &mut out {
            for cell in row {
                *cell *= s;
            }
        }
        Mat3(out)
    }
}

/// A quaternion `w + xi + yj + zk`; orientation quaternions are kept
/// normalized by the owning setters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quat {
    pub w: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Quat {
    pub fn new(w: f64, x: f64, y: f64, z: f64) -> Self {
        Self { w, x, y, z }
    }

    pub fn identity() -> Self {
        Self::new(1.0, 0.0, 0.0, 0.0)
    }

    /// A pure quaternion `(0, v)` — used for `dq/dt = 1/2 (0, w) * q`.
    pub fn pure(v: Vec3) -> Self {
        Self::new(0.0, v.x, v.y, v.z)
    }

    pub fn from_axis_angle(axis: Vec3, angle: f64) -> Self {
        let a = axis.normalize();
        let (s, c) = (angle * 0.5).sin_cos();
        Self::new(c, a.x * s, a.y * s, a.z * s)
    }

    pub fn norm(self) -> f64 {
        (self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Unit quaternion in the same direction; degenerate input
    /// normalizes to the identity.
    pub fn normalize(self) -> Quat {
        let n = self.norm();
        if n > 0.0 && n.is_finite() {
            Quat::new(self.w / n, self.x / n, self.y / n, self.z / n)
        } else {
            Quat::identity()
        }
    }

    pub fn conjugate(self) -> Quat {
        Quat::new(self.w, -self.x, -self.y, -self.z)
    }

    /// Inverse of a unit quaternion (its conjugate).
    pub fn inverse(self) -> Quat {
        self.conjugate()
    }

    /// Rotates a vector by this (unit) quaternion: `q v q*`.
    pub fn rotate(self, v: Vec3) -> Vec3 {
        let qv = Vec3::new(self.x, self.y, self.z);
        let t = 2.0 * qv.cross(v);
        v + self.w * t + qv.cross(t)
    }

    /// The equivalent rotation matrix of this (unit) quaternion.
    pub fn to_rotation_matrix(self) -> Mat3 {
        let (w, x, y, z) = (self.w, self.x, self.y, self.z);
        Mat3([
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - w * z),
                2.0 * (x * z + w * y),
            ],
            [
                2.0 * (x * y + w * z),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - w * x),
            ],
            [
                2.0 * (x * z - w * y),
                2.0 * (y * z + w * x),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ])
    }
}

impl Mul for Quat {
    type Output = Quat;
    /// Hamilton product.
    fn mul(self, r: Quat) -> Quat {
        Quat::new(
            self.w * r.w - self.x * r.x - self.y * r.y - self.z * r.z,
            self.w * r.x + self.x * r.w + self.y * r.z - self.z * r.y,
            self.w * r.y - self.x * r.z + self.y * r.w + self.z * r.x,
            self.w * r.z + self.x * r.y - self.y * r.x + self.z * r.w,
        )
    }
}

impl Mul<f64> for Quat {
    type Output = Quat;
    fn mul(self, s: f64) -> Quat {
        Quat::new(self.w * s, self.x * s, self.y * s, self.z * s)
    }
}

impl Add for Quat {
    type Output = Quat;
    fn add(self, r: Quat) -> Quat {
        Quat::new(self.w + r.w, self.x + r.x, self.y + r.y, self.z + r.z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    fn vec_approx(a: Vec3, b: Vec3, tol: f64) -> bool {
        approx(a.x, b.x, tol) && approx(a.y, b.y, tol) && approx(a.z, b.z, tol)
    }

    #[test]
    fn vec3_dot_cross() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        assert_eq!(a.dot(b), 32.0);
        assert_eq!(a.cross(b), Vec3::new(-3.0, 6.0, -3.0));
        assert_eq!(Vec3::zeros().normalize(), Vec3::zeros());
        assert!(approx(Vec3::new(3.0, 4.0, 0.0).norm(), 5.0, 1e-15));
    }

    #[test]
    fn mat3_inverse_roundtrip() {
        let m = Mat3([[2.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 4.0]]);
        let inv = m.try_inverse().expect("invertible");
        let id = m * inv;
        for i in 0..3 {
            for j in 0..3 {
                let expect = if i == j { 1.0 } else { 0.0 };
                assert!(approx(id.0[i][j], expect, 1e-14), "({i},{j}) = {}", id.0[i][j]);
            }
        }
        assert!(Mat3::zeros().try_inverse().is_none());
        let d = Mat3::from_diagonal(Vec3::new(2.0, 0.0, 4.0)).inverse_diagonal();
        assert_eq!(d, Mat3::from_diagonal(Vec3::new(0.5, 0.0, 0.25)));
    }

    #[test]
    fn mat3_add_sub_skew_outer_identities() {
        let a = Vec3::new(1.0, -2.0, 3.0);
        let b = Vec3::new(-4.0, 0.5, 2.0);
        let c = Vec3::new(0.25, 7.0, -1.5);
        // skew(a) * b == a × b, and [a]× is antisymmetric
        assert_eq!(Mat3::skew(a) * b, a.cross(b));
        assert_eq!(Mat3::skew(a).transpose(), Mat3::skew(a) * -1.0);
        // outer(a, b) * c == a * (b·c)
        assert_eq!(Mat3::outer(a, b) * c, a * b.dot(c));
        // (A + B) v == A v + B v ; (A - B) v == A v - B v
        let m1 = Mat3::skew(a);
        let m2 = Mat3::outer(b, c);
        assert!(vec_approx((m1 + m2) * a, m1 * a + m2 * a, 1e-13));
        assert!(vec_approx((m1 - m2) * a, m1 * a - m2 * a, 1e-13));
        // AddAssign matches Add
        let mut m3 = m1;
        m3 += m2;
        assert_eq!(m3, m1 + m2);
    }

    #[test]
    fn quat_rotation_matches_matrix() {
        let q = Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), std::f64::consts::FRAC_PI_2);
        let v = Vec3::new(1.0, 0.0, 0.0);
        assert!(vec_approx(q.rotate(v), Vec3::new(0.0, 1.0, 0.0), 1e-15));
        assert!(vec_approx(q.to_rotation_matrix() * v, Vec3::new(0.0, 1.0, 0.0), 1e-15));
        // q * q^-1 = identity
        let qi = q * q.inverse();
        assert!(approx(qi.w, 1.0, 1e-15) && approx(qi.x, 0.0, 1e-15));
        // arbitrary axis: rotate then rotate back
        let q2 = Quat::from_axis_angle(Vec3::new(1.0, 2.0, -0.5), 0.83);
        let w = Vec3::new(-2.0, 0.25, 7.0);
        assert!(vec_approx(q2.inverse().rotate(q2.rotate(w)), w, 1e-13));
    }
}
