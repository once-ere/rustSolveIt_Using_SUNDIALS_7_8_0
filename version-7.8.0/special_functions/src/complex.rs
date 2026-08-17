//! A minimal complex number, `Complex64`.
//!
//! Quantum wavefunctions are complex by construction, so the
//! Crank–Nicolson propagator in [`crate::tridiag`] needs this. It is
//! deliberately small — arithmetic, conjugation, modulus, and the
//! exponential — rather than a general numeric-tower crate, because the
//! project takes no external dependencies and this is all the solvers
//! require.
//!
//! Written from the definitions; nothing here is derived from any
//! third-party source.

use std::ops::{Add, Div, Mul, Neg, Sub};

/// A double-precision complex number, `re + i*im`.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Complex64 {
    pub re: f64,
    pub im: f64,
}

impl Complex64 {
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }
    /// The real number `re + 0i`.
    pub const fn real(re: f64) -> Self {
        Self { re, im: 0.0 }
    }
    /// The imaginary unit.
    pub const I: Self = Self { re: 0.0, im: 1.0 };
    pub const ZERO: Self = Self { re: 0.0, im: 0.0 };
    pub const ONE: Self = Self { re: 1.0, im: 0.0 };

    pub fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }
    /// `|z|^2`, without the square root — cheaper and exact when you
    /// only need to compare magnitudes or form a probability density.
    pub fn norm_sqr(self) -> f64 {
        self.re * self.re + self.im * self.im
    }
    pub fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }
    pub fn arg(self) -> f64 {
        self.im.atan2(self.re)
    }
    /// `e^z = e^re (cos im + i sin im)`.
    pub fn exp(self) -> Self {
        let m = self.re.exp();
        Self::new(m * self.im.cos(), m * self.im.sin())
    }
    /// Principal natural logarithm, `ln|z| + i arg(z)`.
    ///
    /// The branch cut runs along the negative real axis, where `arg`
    /// jumps from `+pi` to `-pi`: `ln` is discontinuous there and any
    /// function built on it inherits that. `ln(0)` is `-inf` in the real
    /// part, which is the right answer rather than an error.
    ///
    /// `ln|z|` is taken through `hypot`, so it does not overflow for
    /// large `z` or underflow for small — the same care `inv` needs.
    pub fn ln(self) -> Self {
        Self::new(self.abs().ln(), self.arg())
    }

    /// `z^p` for a real exponent, as `exp(p ln z)` — so it inherits
    /// `ln`'s branch cut along the negative real axis.
    ///
    /// `0^p` is `0` for positive `p` and infinite otherwise, which is
    /// the limit rather than the NaN the naive `exp(p * ln 0)` gives.
    pub fn powf(self, p: f64) -> Self {
        if self.re == 0.0 && self.im == 0.0 {
            return if p > 0.0 {
                Self::ZERO
            } else if p == 0.0 {
                Self::ONE
            } else {
                Self::new(f64::INFINITY, 0.0)
            };
        }
        (self.ln() * p).exp()
    }

    /// `z^p` for a **complex** exponent, as `exp(p ln z)`.
    ///
    /// Same branch cut as [`Self::ln`]. Note that with a complex
    /// exponent the modulus of the result depends on `arg z` as well as
    /// `|z|` — `i^i` is real, and about `0.2079` — so a branch choice
    /// here is not a phase convention, it changes the magnitude.
    pub fn powc(self, p: Self) -> Self {
        if self.re == 0.0 && self.im == 0.0 {
            return if p.re > 0.0 {
                Self::ZERO
            } else if p.re == 0.0 && p.im == 0.0 {
                Self::ONE
            } else {
                Self::new(f64::INFINITY, 0.0)
            };
        }
        (self.ln() * p).exp()
    }

    /// `sin z`, from the real definition
    /// `sin(x+iy) = sin x cosh y + i cos x sinh y`.
    ///
    /// Built this way rather than from `exp` so that it stays accurate
    /// for small `|y|`, where `(e^{iz} - e^{-iz})/2i` cancels.
    pub fn sin(self) -> Self {
        Self::new(
            self.re.sin() * self.im.cosh(),
            self.re.cos() * self.im.sinh(),
        )
    }

    /// `cos z = cos x cosh y - i sin x sinh y`.
    pub fn cos(self) -> Self {
        Self::new(
            self.re.cos() * self.im.cosh(),
            -self.re.sin() * self.im.sinh(),
        )
    }

    /// `e^{i*theta}` — the common case in a propagator.
    pub fn from_polar(r: f64, theta: f64) -> Self {
        Self::new(r * theta.cos(), r * theta.sin())
    }
    pub fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }
    /// Reciprocal, by **Smith's algorithm** — dividing through by the
    /// larger component so `re^2 + im^2` is never formed.
    ///
    /// The naive `conj / norm_sqr` looks equivalent and is not. At
    /// `|z| ~ 1e-199` the squared modulus underflows to zero and the
    /// reciprocal comes back NaN; at `|z| ~ 1e199` it overflows and the
    /// reciprocal comes back zero. Both are inside the range of ordinary
    /// f64 values.
    ///
    /// This was a real defect, not a hypothetical: the complex Bessel
    /// routine normalises by a sum that legitimately reaches 1e-199, and
    /// every value it returned was NaN. The doc comment here previously
    /// claimed the scaling was being done when it was not — which is the
    /// worse half of the bug, because it invited trusting it.
    pub fn inv(self) -> Self {
        if self.re == 0.0 && self.im == 0.0 {
            return Self::new(f64::INFINITY, f64::INFINITY);
        }
        if self.re.abs() >= self.im.abs() {
            let r = self.im / self.re;
            let d = self.re + self.im * r;
            Self::new(1.0 / d, -r / d)
        } else {
            let r = self.re / self.im;
            let d = self.re * r + self.im;
            Self::new(r / d, -1.0 / d)
        }
    }
}

impl Add for Complex64 {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Self::new(self.re + o.re, self.im + o.im)
    }
}
impl Sub for Complex64 {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        Self::new(self.re - o.re, self.im - o.im)
    }
}
impl Mul for Complex64 {
    type Output = Self;
    fn mul(self, o: Self) -> Self {
        Self::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }
}
impl Mul<f64> for Complex64 {
    type Output = Self;
    fn mul(self, s: f64) -> Self {
        Self::new(self.re * s, self.im * s)
    }
}
impl Div for Complex64 {
    type Output = Self;
    // clippy flags `*` inside a `Div` impl as a likely copy-paste slip.
    // Here it is the definition: z/w = z * (1/w), and routing through
    // `inv` means the overflow-avoiding scaling lives in exactly one
    // place instead of being duplicated.
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, o: Self) -> Self {
        self * o.inv()
    }
}
impl Neg for Complex64 {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.re, -self.im)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arithmetic_identities() {
        let a = Complex64::new(3.0, -4.0);
        let b = Complex64::new(-1.0, 2.0);
        assert_eq!(a + b, Complex64::new(2.0, -2.0));
        assert_eq!(a - b, Complex64::new(4.0, -6.0));
        // (3-4i)(-1+2i) = -3 + 6i + 4i - 8i^2 = 5 + 10i
        assert_eq!(a * b, Complex64::new(5.0, 10.0));
        assert_eq!(a.abs(), 5.0);
        assert_eq!(a.norm_sqr(), 25.0);
        // z * conj(z) = |z|^2
        let p = a * a.conj();
        assert!((p.re - 25.0).abs() < 1e-14 && p.im.abs() < 1e-14);
        // z / z = 1
        let q = a / a;
        assert!((q.re - 1.0).abs() < 1e-15 && q.im.abs() < 1e-15);
        // i^2 = -1
        let ii = Complex64::I * Complex64::I;
        assert_eq!(ii, Complex64::new(-1.0, 0.0));
    }

    /// Real powers must agree with elementary cases and compose.
    #[test]
    fn real_powers() {
        let z = Complex64::new(1.3, -0.7);
        // z^1 = z, z^0 = 1
        assert!((z.powf(1.0) - z).abs() < 1e-14);
        assert!((z.powf(0.0) - Complex64::ONE).abs() < 1e-14);
        // z^2 = z*z
        assert!((z.powf(2.0) - z * z).abs() < 1e-13);
        // z^0.5 squared is z again
        let r = z.powf(0.5);
        assert!((r * r - z).abs() < 1e-13, "sqrt(z)^2 = {:?}", r * r);
        // z^-1 = 1/z
        assert!((z.powf(-1.0) - z.inv()).abs() < 1e-13);
        // a positive real base behaves like the real power
        let p = Complex64::real(3.0).powf(1.7);
        assert!((p.re - 3.0_f64.powf(1.7)).abs() < 1e-12 && p.im.abs() < 1e-12);
        // 0^p: the limit, not NaN
        assert_eq!(Complex64::ZERO.powf(2.0), Complex64::ZERO);
        assert_eq!(Complex64::ZERO.powf(0.0), Complex64::ONE);
        assert!(Complex64::ZERO.powf(-1.0).re.is_infinite());
    }

    /// `ln` must invert `exp`, and must survive magnitudes where a
    /// naive `sqrt(re^2 + im^2)` would overflow or underflow.
    #[test]
    fn logarithm_inverts_the_exponential() {
        for z in [
            Complex64::new(1.0, 0.0),
            Complex64::new(0.3, -2.0),
            Complex64::new(-1.5, 0.7),
            Complex64::new(1e-200, 1e-200),
            Complex64::new(1e200, -1e200),
        ] {
            let l = z.ln();
            assert!(l.is_finite(), "ln({z:?}) = {l:?}");
            // exp(ln z) = z, relative to |z|
            let back = l.exp();
            let err = (back - z).abs() / z.abs();
            assert!(err < 1e-13, "exp(ln {z:?}) = {back:?}, relative error {err}");
        }
        // ln of a positive real is real
        let l = Complex64::real(7.0).ln();
        assert!((l.re - 7.0_f64.ln()).abs() < 1e-15 && l.im.abs() < 1e-15);
        // ln(-1) = i pi: the branch cut is approached from above
        let l = Complex64::real(-1.0).ln();
        assert!(l.re.abs() < 1e-15 && (l.im - std::f64::consts::PI).abs() < 1e-15);
        // ln(0) is -inf, not NaN
        assert!(Complex64::ZERO.ln().re.is_infinite());
    }

    /// Reciprocal and division must survive magnitudes where the
    /// squared modulus underflows or overflows — well inside the range
    /// of ordinary f64 values.
    #[test]
    fn reciprocal_survives_extreme_magnitudes() {
        for &m in &[1e-199_f64, 1e-160, 1.0, 1e160, 1e199] {
            for z in [
                Complex64::new(m, 0.0),
                Complex64::new(0.0, m),
                Complex64::new(m, m),
                Complex64::new(-m, 0.3 * m),
            ] {
                let inv = z.inv();
                assert!(inv.is_finite(), "1/{z:?} = {inv:?}");
                // z * (1/z) must be 1
                let one = z * inv;
                assert!(
                    (one.re - 1.0).abs() < 1e-12 && one.im.abs() < 1e-12,
                    "z * (1/z) = {one:?} for z = {z:?}"
                );
                // and division agrees
                let q = z / z;
                assert!(
                    (q.re - 1.0).abs() < 1e-12 && q.im.abs() < 1e-12,
                    "z / z = {q:?} for z = {z:?}"
                );
            }
        }
        // 1/0 is infinite, not NaN
        let inv0 = Complex64::ZERO.inv();
        assert!(inv0.re.is_infinite() && inv0.im.is_infinite());
    }

    #[test]
    fn eulers_identity_and_exp() {
        // e^{i pi} + 1 = 0
        let z = (Complex64::I * std::f64::consts::PI).exp();
        assert!((z.re + 1.0).abs() < 1e-15 && z.im.abs() < 1e-15);
        // |e^{i t}| = 1 for any t
        for &t in &[0.0, 0.3, 1.7, -2.9, 10.0] {
            assert!((Complex64::from_polar(1.0, t).abs() - 1.0).abs() < 1e-15);
        }
        // e^{a+b} = e^a e^b
        let a = Complex64::new(0.4, 1.1);
        let b = Complex64::new(-0.7, 0.5);
        let l = (a + b).exp();
        let r = a.exp() * b.exp();
        assert!((l.re - r.re).abs() < 1e-14 && (l.im - r.im).abs() < 1e-14);
    }
}
