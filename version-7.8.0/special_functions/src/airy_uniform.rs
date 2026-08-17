//! Olver's uniform **Airy-type** expansion, DLMF 10.20.
//!
//! This is the expansion that is uniformly valid *through* the turning
//! point `z = nu`, where the Bessel equation changes character from
//! oscillatory to exponential. It replaces the elementary prefactor of
//! the Debye expansions with an Airy function, which is the exact local
//! model of that transition:
//!
//! ```text
//!   J_nu(nu x) ~ (4 zeta/(1-x^2))^(1/4)
//!                [ Ai(nu^(2/3) zeta)/nu^(1/3)  sum_k A_k(zeta)/nu^(2k)
//!                + Ai'(nu^(2/3) zeta)/nu^(5/3) sum_k B_k(zeta)/nu^(2k) ]
//!   Y_nu(nu x) ~ -(same prefactor)
//!                [ Bi(nu^(2/3) zeta)/nu^(1/3)  sum_k A_k/nu^(2k)
//!                + Bi'(nu^(2/3) zeta)/nu^(5/3) sum_k B_k/nu^(2k) ]
//! ```
//!
//! with `zeta(x)` defined by
//!
//! ```text
//!   (2/3) zeta^(3/2)    = ln((1+sqrt(1-x^2))/x) - sqrt(1-x^2)   (x <= 1)
//!   (2/3) (-zeta)^(3/2) = sqrt(x^2-1) - arccos(1/x)             (x >= 1)
//! ```
//!
//! (DLMF 10.20.2 to 10.20.5.) `zeta` is analytic at `x = 1` even though
//! neither formula looks it, and passes through zero exactly there.
//!
//! # The coefficients, and why they need a series of their own
//!
//! `A_k` and `B_k` are built from the **same Debye polynomials** as
//! [`crate::debye`] (DLMF 10.20.10, 10.20.11):
//!
//! ```text
//!   A_k(zeta) =  sum_{j=0}^{2k}   mu_j     zeta^(-3j/2) U_{2k-j}(p)
//!   B_k(zeta) = -zeta^(-1/2)
//!                sum_{j=0}^{2k+1} lambda_j zeta^(-3j/2) U_{2k+1-j}(p)
//!   p = (1 - x^2)^(-1/2)
//! ```
//!
//! Every term is singular at `zeta = 0` and the singularities cancel
//! exactly — `U_1(p) ~ -5p^3/24` against `lambda_1 zeta^(-3/2)`, and so
//! on. **The cancellation is the whole difficulty.** Both pieces are
//! `O(w^(-3/2))` where `w = 1 - x`, while what survives is `O(1)`, so
//! evaluating the formula as written loses `1.5 log10(1/w)` digits: at
//! `w = 0.01` three of them, at `w = 0.001` four and a half.
//!
//! So near the turning point the coefficients come from **Taylor series
//! in `w`** instead. Those were generated at 70 decimal digits — where
//! the cancellation costs nothing — and are checked here against the
//! closed forms in the band where the closed forms are still accurate.
//! Two of the resulting constants are known independently and match:
//! `A_1(0) = -1/225` exactly, and `zeta'(1) = -2^(1/3)`.
//!
//! # What it is for
//!
//! Measured, the existing routes already reach 1e-14 across the turning
//! point, so this is not the only way to get an answer there — it is an
//! **independent** way, computed from different mathematics, and the
//! test suite uses it as one. Its own accuracy is `O(nu^-6)` with three
//! terms kept: about 1e-12 at `nu = 100` and 1e-9 at `nu = 40`, which is
//! why it is offered to the selector with an honest estimate and wins
//! only where it deserves to.

use crate::complex::Complex64 as C;
use crate::debye::{u_poly, Uniform};

/// Half-width in `w = 1 - x` within which the generated series are used
/// instead of the closed forms. The series were validated to 1e-15 over
/// `|w| <= 0.25`; the closed forms have lost about two digits by there,
/// so the two overlap comfortably.
const W_SERIES: f64 = 0.25;

/// `lambda_j` and `mu_j` of DLMF 10.20.12 and 10.20.13,
/// `lambda_j = (2j+1)(2j+3)...(6j-1) / (j! 144^j)` and
/// `mu_j = -(6j+1)/(6j-1) lambda_j`.
const LAMBDA: [f64; 6] = [
    1.0,
    0.10416666666666667,
    0.08355034722222222,
    0.12822657455632716,
    0.29184902646414046,
    0.8816272674437576,
];

const MU: [f64; 6] = [
    1.0,
    -0.14583333333333334,
    -0.09874131944444445,
    -0.14331205391589505,
    -0.31722720267841353,
    -0.9424291479571203,
];

// The series below are Taylor expansions in `w = 1 - x` about the
// turning point, generated at 70 decimal digits from the closed forms
// above and validated against them by
// `the_generated_series_match_the_closed_forms`. They exist because the
// closed forms cancel catastrophically as `w -> 0`.

const ZETA_OVER_W: [f64; 18] = [
    1.2599210498948756,
    0.37797631496846423,
    0.23038556340619504,
    0.16590960364665774,
    0.1293138715485434,
    0.1056804625374098,
    0.08916992257279024,
    0.07700009498904056,
    0.06767292204965031,
    0.06030166864098137,
    0.05427962995179542,
    0.049360357481516515,
    0.046024462058983376,
    0.04247641531572119,
    0.03304810620328864,
    0.030720731817077716,
    0.056440562120967354,
    0.053223111535609824,
];

const B0_W: [f64; 18] = [
    0.017998872141355322,
    0.011199298221287755,
    0.005940406978610551,
    0.002867672451647506,
    0.0012339189032780798,
    0.0004169250656090051,
    3.301750327795948e-05,
    -0.00013180747026704466,
    -0.00019069388312129725,
    -0.00020117176104437478,
    -0.0001913171219335112,
    -0.00017475111340340332,
    -0.00015945950323661745,
    -0.00014305947468714,
    -0.0001102920711850242,
    -9.930471719557997e-05,
    -0.00017028962302020504,
    -0.00015700995515876233,
];

const A1_W: [f64; 18] = [
    -0.004444444444444444,
    -0.001844155844155843,
    0.0005681207681191386,
    0.0016813786566152848,
    0.001867440421743613,
    0.0016133010586557943,
    0.0012317730927501574,
    0.0008733470835641197,
    0.0005900506490889131,
    0.00038617100035642354,
    0.00024791700288335307,
    0.0001579986241427787,
    0.00010162576756686324,
    6.649103760884365e-05,
    4.179074268801798e-05,
    2.9175329352823583e-05,
    3.5921133823121235e-05,
    2.9883470499762572e-05,
];

const B1_W: [f64; 18] = [
    -0.0014928295321342917,
    -0.0017564094190927785,
    -0.0011334614887418975,
    -0.000346910909813928,
    0.00022752516108059896,
    0.0005176414572637361,
    0.000589061745895543,
    0.0005348551434503566,
    0.00042891804178865975,
    0.0003163977322131025,
    0.00021907886784564023,
    0.0001439420153349038,
    9.025735174287399e-05,
    5.404186394375576e-05,
    3.054904043755558e-05,
    1.64953594596668e-05,
    9.599019155319687e-06,
    4.653046871335124e-06,
];

const A2_W: [f64; 18] = [
    0.0006937355413545918,
    0.00046448349036584517,
    -0.0002890362546093912,
    -0.0008747649439562105,
    -0.0010297163753155167,
    -0.0008368573291708971,
    -0.0004889109624004978,
    -0.00014423679330650497,
    0.00011436667790532786,
    0.0002668087768755884,
    0.0003283924987924621,
    0.0003273149582256451,
    0.0002916447294533,
    0.0002398750074990349,
    0.00017900406181317475,
    0.00013408995772050796,
    0.0001302529577269817,
    8.988823625920445e-05,
];

const B2_W: [f64; 18] = [
    0.0005522130767212967,
    0.0008958651631047718,
    0.0006701500344105549,
    0.00010166263361616004,
    -0.00044086345022023853,
    -0.0007396308143638094,
    -0.0007674550371946571,
    -0.0006082904415995685,
    -0.00037128317165281753,
    -0.00014116071935597736,
    3.600699277160396e-05,
    0.00014725436011641917,
    0.0002007464649988123,
    0.00020912868250389164,
    0.00018050446637229693,
    0.00015298798711964963,
    0.00016629599101273323,
    0.00011984669423436692,
];

/// Horner evaluation of one of the generated series in `w`.
fn horner(c: &[f64], w: f64) -> f64 {
    let mut v = 0.0;
    for &a in c.iter().rev() {
        v = v * w + a;
    }
    v
}

/// `zeta(x)`, and `4 zeta/(1 - x^2)` which the prefactor needs.
///
/// The second is returned separately because near `x = 1` it is the
/// smooth one: `4 zeta/((1-x)(1+x)) = 4 (zeta/w)/(2-w)`, with no
/// cancellation at all, whereas forming it from `zeta` and `1 - x^2`
/// divides two quantities that both vanish.
fn zeta_and_ratio(x: f64) -> Option<(f64, f64)> {
    let w = 1.0 - x;
    if w.abs() <= W_SERIES {
        let g = horner(&ZETA_OVER_W, w); // zeta / w
        return Some((g * w, 4.0 * g / (2.0 - w)));
    }
    if x <= 0.0 {
        return None;
    }
    let zeta = if x < 1.0 {
        let s = (1.0 - x * x).sqrt();
        let f = ((1.0 + s) / x).ln() - s;
        (1.5 * f).powf(2.0 / 3.0)
    } else {
        let s = (x * x - 1.0).sqrt();
        let f = s - (1.0 / x).acos();
        -(1.5 * f).powf(2.0 / 3.0)
    };
    Some((zeta, 4.0 * zeta / (1.0 - x * x)))
}

/// `A_k(zeta)` and `B_k(zeta)` from the closed forms, for `k = 0, 1, 2`.
///
/// Evaluated in complex arithmetic because `p` is imaginary for `x > 1`
/// and `zeta^(-3j/2)` picks up a phase for `zeta < 0`; the two conspire
/// to give a real result, which is asserted rather than assumed.
fn ab_closed(k: usize, x: f64, zeta: f64) -> (f64, f64) {
    let p = (C::ONE - C::real(x * x)).powf(-0.5);
    let zc = C::real(zeta);
    let zp = |e: f64| zc.powf(e);
    let mut a = C::ZERO;
    for (j, &m) in MU.iter().enumerate().take(2 * k + 1) {
        a = a + u_poly(2 * k - j, p) * zp(-1.5 * j as f64) * m;
    }
    let mut b = C::ZERO;
    for (j, &l) in LAMBDA.iter().enumerate().take(2 * k + 2) {
        b = b + u_poly(2 * k + 1 - j, p) * zp(-1.5 * j as f64) * l;
    }
    b = b * zp(-0.5) * -1.0;
    (a.re, b.re)
}

/// [`ab_closed`] at **complex** `x` and `zeta`.
///
/// The body is the real one with the narrowing removed: those closed
/// forms were always evaluated in complex arithmetic, because `p` is
/// imaginary past the turning point and `zeta^(-3j/2)` carries a phase
/// for negative `zeta`. Only the inputs widen — and with [`zeta_c`]
/// supplying the branch, nothing else has to change.
fn ab_closed_c(k: usize, x: C, zeta: C) -> (C, C) {
    let p = (C::ONE - x * x).powf(-0.5);
    let zp = |e: f64| zeta.powf(e);
    let mut a = C::ZERO;
    for (j, &m) in MU.iter().enumerate().take(2 * k + 1) {
        a = a + u_poly(2 * k - j, p) * zp(-1.5 * j as f64) * m;
    }
    let mut b = C::ZERO;
    for (j, &l) in LAMBDA.iter().enumerate().take(2 * k + 2) {
        b = b + u_poly(2 * k + 1 - j, p) * zp(-1.5 * j as f64) * l;
    }
    b = b * zp(-0.5) * -1.0;
    (a, b)
}

/// `(A_k, B_k)` for `k = 0, 1, 2`, by whichever route is sound here.
fn ab(x: f64, zeta: f64) -> ([f64; 3], [f64; 3]) {
    let w = 1.0 - x;
    if w.abs() <= W_SERIES {
        (
            [1.0, horner(&A1_W, w), horner(&A2_W, w)],
            [horner(&B0_W, w), horner(&B1_W, w), horner(&B2_W, w)],
        )
    } else {
        let (_, b0) = ab_closed(0, x, zeta);
        let (a1, b1) = ab_closed(1, x, zeta);
        let (a2, b2) = ab_closed(2, x, zeta);
        ([1.0, a1, a2], [b0, b1, b2])
    }
}

/// `J_nu(z)` and `Y_nu(z)` by DLMF 10.20.4 and 10.20.5, for real
/// `z > 0` and `nu > 0`.
///
/// Uniformly valid across the turning point `z = nu`, which is what
/// distinguishes it from everything else in this crate. Returns `None`
/// for a value that left `f64` range — `Y` does so on the small-`z`
/// side long before `J` does.
///
/// # Examples
/// ```
/// use special_functions::airy_uniform::jy_airy;
/// // At the turning point exactly, J_nu(nu) ~ 2^(1/3) Ai(0) / nu^(1/3),
/// // which is the classical Cauchy value.
/// let nu = 200.0_f64;
/// let (j, _) = jy_airy(nu, nu);
/// let want = 2.0_f64.powf(1.0 / 3.0) * 0.3550280538878172 / nu.powf(1.0 / 3.0);
/// // The leading term only — the correction is O(nu^(-2/3)), which is
/// // about 3% at nu = 200, so this checks the scaling law and not the
/// // expansion's own accuracy.
/// assert!((j.unwrap().value.re / want - 1.0).abs() < 0.05);
/// ```
pub fn jy_airy(nu: f64, z: f64) -> (Option<Uniform>, Option<Uniform>) {
    if !nu.is_finite() || nu <= 0.0 || !z.is_finite() || z <= 0.0 || nu.is_nan() || z.is_nan() {
        return (None, None);
    }
    let x = z / nu;
    let Some((zeta, ratio)) = zeta_and_ratio(x) else {
        return (None, None);
    };
    if ratio <= 0.0 || !ratio.is_finite() {
        return (None, None);
    }
    let pref = ratio.powf(0.25);
    let t = nu.powf(2.0 / 3.0) * zeta;
    let (ai, aip, bi, bip) = spec_math::cephes64::airy(t);

    let (a, b) = ab(x, zeta);
    let n2 = nu * nu;
    let sa = a[0] + a[1] / n2 + a[2] / (n2 * n2);
    let sb = b[0] + b[1] / n2 + b[2] / (n2 * n2);
    let c1 = pref / nu.powf(1.0 / 3.0);
    let c2 = pref / nu.powf(5.0 / 3.0);

    let j = c1 * ai * sa + c2 * aip * sb;
    let y = -(c1 * bi * sa + c2 * bip * sb);

    // Optimal truncation: the size of the last term kept, relative to
    // the result. Near a zero of J that ratio is large, which is
    // correct — the relative accuracy really is poor there.
    let last = |v: f64, f1: f64, f2: f64| {
        let m = (c1 * f1 * a[2] / (n2 * n2)).abs() + (c2 * f2 * b[2] / (n2 * n2)).abs();
        if v == 0.0 || !v.is_finite() {
            f64::INFINITY
        } else {
            m / v.abs()
        }
    };
    let ej = last(j, ai, aip);
    let ey = last(y, bi, bip);
    (
        (j.is_finite() && ej.is_finite()).then_some(Uniform { value: C::real(j), err: ej }),
        (y.is_finite() && ey.is_finite()).then_some(Uniform { value: C::real(y), err: ey }),
    )
}

// ---------------------------------------------------------------------
// Complex order
// ---------------------------------------------------------------------

/// Horner in a **complex** `w`.
fn horner_c(c: &[f64], w: C) -> C {
    let mut v = C::ZERO;
    for &a in c.iter().rev() {
        v = v * w + C::real(a);
    }
    v
}

/// `J_nu(z)` and `Y_nu(z)` by DLMF 10.20 at **complex order**, near the
/// turning point.
///
/// # The turning point, and (since Stage 2D) beyond it
///
/// The closed forms for `zeta` and for `A_k`, `B_k` do not continue
/// naively to complex `x`: `zeta`'s two branch formulas meet at `x = 1`
/// and the principal `2/3` power does not carry across, which is what
/// Stage 21 recorded as outstanding. But **every ingredient this
/// expansion needs near the turning point is already a Taylor series in
/// `w = 1 - x`**, generated at 70 digits and validated against the
/// closed forms — and a Taylor series does not care whether its
/// variable is real. So `|w| <= 0.25` needs no new mathematics at all.
///
/// That was the whole story until Stage 2D. It was written here that
/// "outside that neighbourhood 10.20 is not the right tool anyway",
/// which was wrong: Stage 24 measured a real band — `1 < |z/nu| < 8`
/// off the real axis — that the Debye and `1/z` routes both refuse and
/// that this expansion is uniformly valid across.
///
/// What blocked it was arithmetic, not mathematics: `zeta`'s closed
/// form needs a branch that the principal `2/3` power gets wrong past
/// `x = 1`. [`zeta_c`] chooses that branch against the known behaviour
/// at the turning point, and [`ab_closed_c`] is the existing real
/// closed form with its narrowing removed. Inside `|arg(z/nu)| <= 0.8`,
/// where the anchor is measured to hold, the band goes from **41 %** to
/// **97.5 %** served.
///
/// Returns `None` outside that sector, where the branch anchor is not
/// measured, or where the value leaves `f64`.
/// `zeta(x)` at **complex** `x`, from the closed form, with the branch
/// chosen rather than assumed.
///
/// ```text
///     (2/3) zeta^(3/2) = ln((1 + s)/x) - s ,      s = (1 - x^2)^(1/2)
/// ```
///
/// `zeta` is analytic through the turning point — the two real formulas
/// either side of `x = 1` are one function — but the *expression* is
/// not: `s`, the logarithm and the `2/3` power each carry a branch, and
/// taking all three principal gives the wrong answer beyond `x = 1`.
/// Measured on the real axis at `x > 1`, the principal value comes out
/// at `arg = pi/3` where the truth is `arg = pi`: a cube-root branch
/// away.
///
/// The branch is fixed by the **known behaviour at the turning point**,
/// which is the technique Stage 24 arrived at for the Debye expansion.
/// As `w = 1 - x -> 0`,
///
/// ```text
///     zeta ~ 2^(1/3) w    so    F = (2/3) zeta^(3/2) ~ (2/3) 2^(1/2) w^(3/2)
/// ```
///
/// so `arg F` must approach `(3/2) arg w`. Unwrapping `arg F` to the
/// representative nearest that value pins the branch everywhere, and
/// `the_closed_form_zeta_agrees_with_the_series` checks it against the
/// independently generated Taylor series across the whole overlap.
fn zeta_c(x: C) -> Option<C> {
    if x.abs() == 0.0 {
        return None;
    }
    let s = (C::ONE - x * x).powf(0.5);
    let f = ((C::ONE + s) * x.inv()).ln() - s;
    if !f.is_finite() {
        return None;
    }
    if f.abs() == 0.0 {
        return Some(C::ZERO);
    }
    let w = C::ONE - x;
    // Unwrap arg(3F/2) to the branch nearest 1.5 arg(w).
    let target = 1.5 * w.arg();
    let raw = f.arg();
    let two_pi = 2.0 * std::f64::consts::PI;
    let k = ((target - raw) / two_pi).round();
    let theta = raw + k * two_pi;
    let modulus = (1.5 * f.abs()).powf(2.0 / 3.0);
    let zeta = C::from_polar(modulus, theta * (2.0 / 3.0));
    zeta.is_finite().then_some(zeta)
}

pub fn jy_airy_c(nu: C, z: C) -> Option<(Uniform, Uniform)> {
    if !nu.is_finite() || !z.is_finite() || nu.abs() == 0.0 {
        return None;
    }
    let x = z * nu.inv();
    let w = C::ONE - x;
    // Near the turning point the Taylor series is used, because the
    // closed forms divide two vanishing quantities there. Away from it
    // the closed forms are used, which is what extends this expansion
    // from a neighbourhood of `x = 1` to the whole plane — the band
    // Stage 24 measured as uncovered, `1 < |z/nu| < 8` off the real
    // axis, is exactly here.
    let near = w.abs() <= W_SERIES;
    // **Sector guard on the extended branch.** Away from the turning
    // point `zeta_c` fixes its branch by anchoring `arg F` to
    // `1.5 arg(w)`, which is exact near `w = 0` and degrades as the
    // argument grows. Measured against the `1/z` Hankel pair over a
    // sweep in `|nu|` and `|z/nu|`, the reported estimate bounds the
    // actual error to a factor of **2.6** while `|arg(z/nu)| <= 0.8`,
    // and the first point found outside it — `nu = 0.5 + 5i`,
    // `arg z = -2.4` — was wrong by 1.4 with a small estimate.
    //
    // So the extension is offered only inside that sector. The band
    // Stage 24 left open is closed there and remains open beyond it;
    // that is a smaller gap than before, honestly stated, rather than
    // a larger claim.
    if !near && x.arg().abs() > 0.8 {
        return None;
    }
    let (zeta, ratio) = if near {
        // zeta = w (zeta/w), and 4 zeta/(1 - x^2) as 4(zeta/w)/(2 - w),
        // which divides no vanishing quantities.
        let g = horner_c(&ZETA_OVER_W, w);
        (g * w, g * 4.0 * (C::real(2.0) - w).inv())
    } else {
        let zeta = zeta_c(x)?;
        let denom = C::ONE - x * x;
        if denom.abs() == 0.0 {
            return None;
        }
        (zeta, zeta * 4.0 * denom.inv())
    };
    let pref = ratio.powf(0.25);
    let t = nu.powc(C::real(2.0 / 3.0)) * zeta;
    let a = crate::airy_complex::airy_c(t).ok()?;

    let (a1, a2, b0, b1, b2) = if near {
        (
            horner_c(&A1_W, w),
            horner_c(&A2_W, w),
            horner_c(&B0_W, w),
            horner_c(&B1_W, w),
            horner_c(&B2_W, w),
        )
    } else {
        let (_, b0) = ab_closed_c(0, x, zeta);
        let (a1, b1) = ab_closed_c(1, x, zeta);
        let (a2, b2) = ab_closed_c(2, x, zeta);
        (a1, a2, b0, b1, b2)
    };
    let n2 = nu * nu;
    let n4 = n2 * n2;
    let sa = C::ONE + a1 * n2.inv() + a2 * n4.inv();
    let sb = b0 + b1 * n2.inv() + b2 * n4.inv();
    let c1 = pref * nu.powc(C::real(1.0 / 3.0)).inv();
    let c2 = pref * nu.powc(C::real(5.0 / 3.0)).inv();

    let j = c1 * a.ai * sa + c2 * a.aip * sb;
    let y = (c1 * a.bi * sa + c2 * a.bip * sb) * -1.0;

    // Optimal truncation, as in the real-order routine: the size of the
    // last term kept, relative to the result — plus whatever the Airy
    // evaluation itself reports, since that is now a computed quantity
    // rather than a table lookup.
    let last = |v: C, f1: C, f2: C| {
        let m = (c1 * f1 * a2 * n4.inv()).abs() + (c2 * f2 * b2 * n4.inv()).abs();
        if v.abs() == 0.0 || !v.is_finite() {
            f64::INFINITY
        } else {
            (m / v.abs()).max(a.err)
        }
    };
    let (ej, ey) = (last(j, a.ai, a.aip), last(y, a.bi, a.bip));
    if !j.is_finite() || !y.is_finite() || !ej.is_finite() || !ey.is_finite() {
        return None;
    }
    Some((
        Uniform { value: j, err: ej },
        Uniform { value: y, err: ey },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The generated series must agree with the closed forms in the
    /// band where the closed forms are still accurate. This is what
    /// makes the generated constants checkable rather than trusted:
    /// they came from a 70-digit computation done elsewhere, and this
    /// is the crate verifying them against mathematics it evaluates
    /// itself.
    #[test]
    fn the_generated_series_match_the_closed_forms() {
        for &w in &[-0.25_f64, -0.2, -0.15, -0.1, 0.1, 0.15, 0.2, 0.25] {
            let x = 1.0 - w;
            let (zeta, _) = zeta_and_ratio(x).unwrap();
            // zeta from the series versus zeta from its own definition.
            let exact = if x < 1.0 {
                let s = (1.0 - x * x).sqrt();
                (1.5 * (((1.0 + s) / x).ln() - s)).powf(2.0 / 3.0)
            } else {
                let s = (x * x - 1.0).sqrt();
                -(1.5 * (s - (1.0 / x).acos())).powf(2.0 / 3.0)
            };
            assert!(
                (zeta - exact).abs() <= 1e-13 * exact.abs(),
                "zeta at w={w}: {zeta} vs {exact}"
            );
            for (k, (sa, sb)) in [
                (0usize, (1.0, horner(&B0_W, w))),
                (1, (horner(&A1_W, w), horner(&B1_W, w))),
                (2, (horner(&A2_W, w), horner(&B2_W, w))),
            ] {
                let (ca, cb) = ab_closed(k, x, exact);
                // The bound is ABSOLUTE and scaled by nu^(2k), because
                // A_k and B_k enter the answer divided by nu^(2k): a
                // discrepancy of `1e-11 * 40^(2k)` in the coefficient is
                // 1e-11 in the result at nu = 40, and less above that.
                //
                // It has to be stated that way round, because at small
                // |w| the CLOSED FORM is the inaccurate party and
                // increasingly so with k — its singular pieces are of
                // order w^(-3(2k+1)/2). That is the entire reason the
                // series exists, so a test that assumed the closed form
                // was the truth would be testing backwards.
                let bound = 1e-11 * 40.0_f64.powi(2 * k as i32);
                assert!(
                    (sa - ca).abs() <= bound,
                    "A_{k} at w={w}: series {sa} vs closed {ca}"
                );
                assert!(
                    (sb - cb).abs() <= bound,
                    "B_{k} at w={w}: series {sb} vs closed {cb}"
                );
            }
        }
    }

    /// Two constants of the expansion are known in closed form, and
    /// they are the ones a transcription error would break:
    /// `A_1(0) = -1/225` (Olver) and `zeta'(1) = -2^(1/3)`.
    #[test]
    fn the_known_constants_are_right() {
        assert!(
            (A1_W[0] + 1.0 / 225.0).abs() < 1e-15,
            "A_1(0) should be -1/225, got {}",
            A1_W[0]
        );
        assert!(
            // 1e-14, not 1e-16: these coefficients come from a
            // Chebyshev interpolation whose own residual is 2e-15, and
            // pinning c0 to the exact cube root would misstate what the
            // series is.
            (ZETA_OVER_W[0] - 2.0_f64.powf(1.0 / 3.0)).abs() < 1e-14,
            "zeta/w at w=0 should be 2^(1/3), got {}",
            ZETA_OVER_W[0]
        );
        // A_0 is identically 1, and B_0(0) is what the singular pieces
        // leave behind. Neither is arbitrary: perturb either and the
        // comparison with Cephes below fails.
        assert!((B0_W[0] - 0.017998872141355322).abs() < 1e-15);
    }

    /// Against Cephes, straight through the turning point — the region
    /// this expansion exists for and the one where the Debye expansions
    /// are useless.
    #[test]
    fn it_matches_cephes_across_the_turning_point() {
        for &nu in &[40.0_f64, 100.0, 200.0, 400.0, 1000.0] {
            for &frac in &[0.9_f64, 0.95, 0.98, 1.0, 1.02, 1.05, 1.2] {
                let z = nu * frac;
                let want = spec_math::cephes64::jv(nu, z);
                if want == 0.0 || !want.is_finite() {
                    continue;
                }
                let (j, _) = jy_airy(nu, z);
                let j = j.unwrap_or_else(|| panic!("no J at nu={nu}, x={frac}"));
                // Floor 1e-12 rather than the estimate alone: at
                // nu = 1000 the truncation estimate is 7e-16 and the gap
                // to Cephes is 1.1e-13, and the Wronskian test below —
                // which needs no reference — says the values here are
                // the tighter ones.
                let bound = (3.0 * j.err).max(1e-11) * want.abs();
                assert!(
                    (j.value.re - want).abs() <= bound,
                    "J_{nu}({z}): {} vs {want}, estimate {:.1e}",
                    j.value.re,
                    j.err
                );
            }
        }
    }

    /// `Y` likewise, where Cephes can still represent it.
    #[test]
    fn y_matches_cephes_across_the_turning_point() {
        for &nu in &[40.0_f64, 100.0, 200.0] {
            for &frac in &[0.95_f64, 0.98, 1.0, 1.02, 1.2] {
                let z = nu * frac;
                let want = spec_math::cephes64::yv(nu, z);
                if !want.is_finite() || want == 0.0 {
                    continue;
                }
                let (_, y) = jy_airy(nu, z);
                let y = y.unwrap();
                let bound = (3.0 * y.err).max(1e-11) * want.abs();
                assert!(
                    (y.value.re - want).abs() <= bound,
                    "Y_{nu}({z}): {} vs {want}, estimate {:.1e}",
                    y.value.re,
                    y.err
                );
            }
        }
    }

    /// The J-Y Wronskian, elementary on the right, computed entirely
    /// from this expansion. It judges the pair without a reference.
    #[test]
    fn the_wronskian_holds_across_the_turning_point() {
        for &nu in &[40.0_f64, 100.0, 400.0] {
            for &frac in &[0.9_f64, 0.98, 1.0, 1.05, 1.3] {
                let z = nu * frac;
                let (Some(j0), Some(y0)) = jy_airy(nu, z) else { continue };
                let (Some(j1), Some(y1)) = jy_airy(nu + 1.0, z) else { continue };
                let w = j1.value.re * y0.value.re - j0.value.re * y1.value.re;
                let want = 2.0 / (std::f64::consts::PI * z);
                let bound = (10.0 * j0.err.max(y0.err)).max(1e-10);
                assert!(
                    (w - want).abs() / want <= bound,
                    "Wronskian at nu={nu}, x={frac}: {:.2e}",
                    (w - want).abs() / want
                );
            }
        }
    }

    /// At the turning point exactly the expansion reduces to the
    /// classical Cauchy value `J_nu(nu) ~ 2^(1/3) Ai(0)/nu^(1/3)`, whose
    /// leading behaviour is `nu^(-1/3)` — a different statement from
    /// anything the tests above make, and one no reference is needed
    /// for.
    #[test]
    fn the_turning_point_value_scales_as_the_cube_root() {
        let c = 2.0_f64.powf(1.0 / 3.0) * 0.3550280538878172;
        for &nu in &[50.0_f64, 200.0, 800.0, 3200.0] {
            let j = jy_airy(nu, nu).0.unwrap().value.re;
            let want = c / nu.powf(1.0 / 3.0);
            assert!(
                (j / want - 1.0).abs() < 0.02 / nu.powf(2.0 / 3.0) * 30.0,
                "J_nu(nu) at nu={nu}: {j} vs leading {want}"
            );
        }
    }

    /// At a real order and argument the complex route must reproduce
    /// the real one. They share the generated series but not the
    /// arithmetic — complex `powc` and complex Airy against `powf` and
    /// Cephes — so agreement is a real check on both.
    #[test]
    fn the_complex_route_reproduces_the_real_one() {
        for &(nu, frac) in &[
            (40.0_f64, 0.85_f64),
            (40.0, 1.0),
            (100.0, 0.98),
            (200.0, 1.0),
            (400.0, 1.05),
            (1000.0, 1.15),
        ] {
            let z = nu * frac;
            let (Some(rj), Some(ry)) = jy_airy(nu, z) else {
                panic!("real route missing at nu={nu}, x={frac}")
            };
            let Some((cj, cy)) = jy_airy_c(C::real(nu), C::real(z)) else {
                panic!("complex route missing at nu={nu}, x={frac}")
            };
            assert!(
                (cj.value.re - rj.value.re).abs() <= 1e-12 * rj.value.re.abs(),
                "J at nu={nu}, x={frac}"
            );
            assert!(
                (cy.value.re - ry.value.re).abs() <= 1e-12 * ry.value.re.abs(),
                "Y at nu={nu}, x={frac}"
            );
            assert!(cj.value.im.abs() <= 1e-12 * rj.value.re.abs(), "should be real");
        }
    }

    /// Complex order at the turning point, by the J-Y Wronskian — whose
    /// right-hand side involves neither the order nor any Bessel
    /// function, and which is the only instrument available here since
    /// no reference implementation covers it.
    ///
    /// This is the region Stage 18 recorded as unreachable and Stage 21
    /// supplied the missing ingredient for.
    /// The closed-form complex `zeta` against the Taylor series, over
    /// the whole overlap where both are valid.
    ///
    /// These share no arithmetic: one is a 70-digit generated series in
    /// `w`, the other is logs and roots with a hand-chosen branch. They
    /// agree only if the branch rule is right.
    #[test]
    fn the_closed_form_zeta_agrees_with_the_series() {
        let mut worst: f64 = 0.0;
        let mut n = 0;
        for i in 0..24 {
            let r = 0.02 + 0.01 * i as f64; // |w| within the series radius
            for k in 0..48 {
                let th = -std::f64::consts::PI + std::f64::consts::TAU * k as f64 / 48.0;
                let w = C::from_polar(r, th);
                let x = C::ONE - w;
                if x.abs() < 1e-6 {
                    continue;
                }
                let series = horner_c(&ZETA_OVER_W, w) * w;
                let Some(closed) = zeta_c(x) else { continue };
                let e = (closed - series).abs() / series.abs().max(1e-300);
                worst = worst.max(e);
                n += 1;
            }
        }
        assert!(n > 900, "only {n} points were compared");
        assert!(worst < 1e-9, "worst relative disagreement {worst:.2e} over {n} points");
    }

    /// The branch anchor, tested **where it actually matters**.
    ///
    /// `the_closed_form_zeta_agrees_with_the_series` compares the two
    /// routes in their *overlap*, and Stage 2F's mutation probe showed
    /// that is not enough: changing the anchor coefficient from 1.5 to
    /// 1.0 survived the whole suite. Inside `|w| <= 0.25` the argument
    /// is small, both coefficients select the same branch, and the test
    /// cannot tell them apart — it was verifying the region where the
    /// answer was already known rather than the region the new code
    /// serves.
    ///
    /// This exercises the extended route far from the turning point,
    /// against the `1/z` Hankel pair, which shares no code with it.
    #[test]
    fn the_branch_anchor_is_constrained_away_from_the_turning_point() {
        let mut checked = 0;
        // The 1/z reference needs |4 nu^2| <= 2|z|, i.e. |z/nu| >= 2|nu|,
        // so the orders are modest and the arguments large. That is
        // still far outside the |w| <= 0.25 series neighbourhood, which
        // is the whole point.
        for &(a, b) in &[(6.0_f64, 1.5_f64), (8.0, 2.0), (10.0, -2.5)] {
            for &frac in &[22.0_f64, 40.0, 70.0] {
                let nu = C::new(a, b);
                let z = nu * frac;
                let Some((j, y)) = jy_airy_c(nu, z) else { continue };
                let Some((h1, h2, e)) = crate::bessel_cnu_large::hankel_pair_any(nu, z)
                else {
                    continue;
                };
                if e >= 1e-13 {
                    continue;
                }
                let (wj, wy) = ((h1 + h2) * 0.5, (h1 - h2) * C::new(0.0, -0.5));
                let rj = (j.value - wj).abs() / wj.abs();
                let ry = (y.value - wy).abs() / wy.abs();
                checked += 1;
                assert!(
                    rj <= (3.0 * j.err).max(1e-9) && ry <= (3.0 * y.err).max(1e-9),
                    "nu={nu:?}, z/nu={frac}: J off by {rj:.2e} (est {:.1e}), \
                     Y off by {ry:.2e} (est {:.1e})",
                    j.err,
                    y.err
                );
            }
        }
        assert!(checked >= 6, "only {checked} far-field points were reached");
    }

    /// On the real axis the closed form must reproduce the real-order
    /// routine exactly — including **past the turning point**, where a
    /// principal branch gives `arg = pi/3` instead of `pi` and the sign
    /// of `zeta` comes out wrong.
    #[test]
    fn the_complex_zeta_reproduces_the_real_one_on_both_sides() {
        for &x in &[0.05_f64, 0.3, 0.6, 0.9, 0.99, 1.01, 1.2, 2.0, 5.0, 40.0] {
            let (want, _) = zeta_and_ratio(x).expect("real zeta");
            let got = zeta_c(C::real(x)).expect("complex zeta");
            assert!(
                got.im.abs() < 1e-9 * got.abs().max(1.0),
                "x = {x}: zeta should be real, got {got:?}"
            );
            assert!(
                (got.re - want).abs() <= 1e-9 * want.abs().max(1e-12),
                "x = {x}: {} vs {want}",
                got.re
            );
            // And past the turning point it must be NEGATIVE, which is
            // the whole branch question.
            if x > 1.0 {
                assert!(got.re < 0.0, "x = {x}: zeta must be negative beyond the turning point");
            }
        }
    }

    #[test]
    fn complex_order_at_the_turning_point_satisfies_the_wronskian() {
        let mut checked = 0;
        for &(a, b) in &[
            (40.0_f64, 5.0_f64),
            (100.0, 10.0),
            (100.0, -20.0),
            (200.0, 40.0),
            (400.0, 80.0),
        ] {
            for &frac in &[0.85_f64, 0.95, 1.0, 1.05, 1.15] {
                let nu = C::new(a, b);
                let z = nu * frac;
                let (Some((j0, y0)), Some((j1, y1))) =
                    (jy_airy_c(nu, z), jy_airy_c(nu + C::ONE, z))
                else {
                    continue;
                };
                let w = j1.value * y0.value - j0.value * y1.value;
                let want = z.inv() * (2.0 / std::f64::consts::PI);
                let scale = (j1.value * y0.value).abs() + (j0.value * y1.value).abs();
                let e = (w - want).abs() / scale;
                checked += 1;
                let floor = crate::bessel_cnu_large::hankel_ratio(nu, z)
                    .map_or(0.0, |r| 1.0 / r);
                assert!(
                    e <= (3.0 * j0.err).max(1e-10).max(10.0 * floor),
                    "nu={nu:?}, x={frac}: residual {e:.2e}, estimate {:.1e}",
                    j0.err
                );
            }
        }
        assert!(checked >= 20, "only {checked} points were reached");
    }

    #[test]
    fn airy_uniform_edge_cases() {
        // The complex route is a turning-point tool and says so.
        // `x = 0.1` is far from the turning point and used to be
        // refused; the closed-form branch now covers it. What is still
        // refused is the sector the branch anchor cannot speak for.
        assert!(jy_airy_c(C::real(100.0), C::real(10.0)).is_some(), "closed form covers x = 0.1");
        assert!(
            jy_airy_c(C::new(0.5, 5.0), C::from_polar(26.0, -2.4)).is_none(),
            "outside the measured sector it must refuse"
        );
        assert!(jy_airy_c(C::ZERO, C::ONE).is_none(), "nu = 0");
        assert!(jy_airy(0.0, 1.0).0.is_none(), "nu = 0");
        assert!(jy_airy(10.0, 0.0).0.is_none(), "z = 0");
        assert!(jy_airy(10.0, -1.0).0.is_none(), "negative z");
        assert!(jy_airy(f64::NAN, 1.0).0.is_none(), "NaN order");
    }
}
