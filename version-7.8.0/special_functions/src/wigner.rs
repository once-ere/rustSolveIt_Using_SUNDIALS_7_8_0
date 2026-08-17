//! Wigner 3-j and 6-j symbols, and Clebsch–Gordan coefficients.
//!
//! These are the algebra of angular-momentum coupling: what you need to
//! add two angular momenta, evaluate a matrix element between coupled
//! states, or recouple three of them.
//!
//! # Conventions
//!
//! Arguments are `f64` because angular momenta are integer *or*
//! half-integer. A value that is neither is rejected rather than
//! rounded — `wigner_3j(0.3, ...)` is a mistake, and quietly answering
//! for `j = 0.5` would hide it.
//!
//! The Condon–Shortley phase convention is used throughout, which is
//! what makes
//! `<j1 m1 j2 m2 | j3 m3> = (-1)^(j1-j2+m3) sqrt(2 j3 + 1) (j1 j2 j3 ; m1 m2 -m3)`
//! the relation between the two.
//!
//! # Algorithm
//!
//! Racah's single-sum formula (DLMF 34.2.4, <https://dlmf.nist.gov/34.2.E4>;
//! Edmonds 1957 §3.6) for the 3-j, and Racah's formula (DLMF 34.4.1,
//! <https://dlmf.nist.gov/34.4.E1>) for the 6-j.
//!
//! Every factorial is taken in **logarithms**. The factorial arguments
//! are all non-negative integers, but they reach `(j1+j2+j3+1)!`, which
//! overflows `f64` past about `j = 85` if formed directly — and the
//! ratio being computed is perfectly finite. Working with `ln n!`
//! throughout keeps the intermediate values `O(1)` in magnitude.
//!
//! # Accuracy note, stated honestly
//!
//! The Racah sum is **alternating**, so it suffers cancellation: for
//! large `j` the result can be many orders of magnitude smaller than the
//! individual terms, and relative accuracy degrades accordingly. This is
//! a property of the formula, not of this implementation, and it is why
//! the tests here check the *orthogonality sums* — which are sensitive
//! to exactly that error — rather than only spot values.

/// `ln(n!)` for a non-negative integer.
///
/// Summed directly rather than via a gamma function: the arguments here
/// are always integers, the counts are small, and this keeps the module
/// free of any dependency on the vendored code so its provenance stays
/// trivially auditable.
fn ln_fact(n: u32) -> f64 {
    (2..=n).map(|k| (k as f64).ln()).sum()
}

/// True when `x` is an integer or a half-integer, within rounding.
fn is_half_integral(x: f64) -> bool {
    x.is_finite() && (2.0 * x - (2.0 * x).round()).abs() < 1e-9
}

/// True when `x` is a whole number, within rounding.
fn is_integral(x: f64) -> bool {
    x.is_finite() && (x - x.round()).abs() < 1e-9
}

/// Round a value known to be integral to `u32`, for use as a factorial
/// argument. Returns `None` if it is negative.
fn idx(x: f64) -> Option<u32> {
    let r = x.round();
    if r < -0.5 {
        None
    } else {
        Some(r as u32)
    }
}

fn check_j(name: &str, label: &str, j: f64) -> Result<(), String> {
    if !is_half_integral(j) {
        return Err(format!(
            "{name}: {label} = {j} is neither an integer nor a half-integer"
        ));
    }
    if j < 0.0 {
        return Err(format!("{name}: {label} = {j} must not be negative"));
    }
    Ok(())
}

fn check_jm(name: &str, jl: &str, ml: &str, j: f64, m: f64) -> Result<(), String> {
    check_j(name, jl, j)?;
    if !is_half_integral(m) {
        return Err(format!(
            "{name}: {ml} = {m} is neither an integer nor a half-integer"
        ));
    }
    // j and m must be of the same kind: j - m is always an integer.
    if !is_integral(j - m) {
        return Err(format!(
            "{name}: {jl} = {j} and {ml} = {m} are incompatible — {jl} - {ml} must be a whole \
             number (both integer, or both half-integer)"
        ));
    }
    Ok(())
}

/// `ln` of the triangle coefficient
/// `Delta(a,b,c) = (a+b-c)! (a-b+c)! (-a+b+c)! / (a+b+c+1)!`.
///
/// Returns `None` when the triangle condition fails, which is the
/// selection rule `|a-b| <= c <= a+b`.
fn ln_delta(a: f64, b: f64, c: f64) -> Option<f64> {
    let p = idx(a + b - c)?;
    let q = idx(a - b + c)?;
    let r = idx(-a + b + c)?;
    let s = idx(a + b + c + 1.0)?;
    Some(ln_fact(p) + ln_fact(q) + ln_fact(r) - ln_fact(s))
}

/// The Wigner 3-j symbol
/// `( j1 j2 j3 ; m1 m2 m3 )`.
///
/// Returns `0.0` whenever a selection rule is violated — that is the
/// mathematically correct value, not an error. Errors are reserved for
/// arguments that are not angular momenta at all.
///
/// # Errors
/// A `j` or `m` that is neither integer nor half-integer, a negative
/// `j`, or a `j`/`m` pair of mismatched kind.
///
/// # Examples
/// ```
/// use special_functions::wigner::wigner_3j;
/// // (1 1 0 ; 0 0 0) = -1/sqrt(3)
/// let v = wigner_3j(1.0, 1.0, 0.0, 0.0, 0.0, 0.0).unwrap();
/// assert!((v + 1.0 / 3.0_f64.sqrt()).abs() < 1e-14);
/// // m1 + m2 + m3 != 0 vanishes by selection rule, and is NOT an error
/// assert_eq!(wigner_3j(1.0, 1.0, 2.0, 1.0, 0.0, 0.0).unwrap(), 0.0);
/// ```
pub fn wigner_3j(
    j1: f64,
    j2: f64,
    j3: f64,
    m1: f64,
    m2: f64,
    m3: f64,
) -> Result<f64, String> {
    const NAME: &str = "wigner_3j";
    check_jm(NAME, "j1", "m1", j1, m1)?;
    check_jm(NAME, "j2", "m2", j2, m2)?;
    check_jm(NAME, "j3", "m3", j3, m3)?;

    // ---- selection rules: all of these give exactly zero -----------
    if (m1 + m2 + m3).abs() > 1e-9 {
        return Ok(0.0);
    }
    if m1.abs() > j1 + 1e-9 || m2.abs() > j2 + 1e-9 || m3.abs() > j3 + 1e-9 {
        return Ok(0.0);
    }
    if j3 < (j1 - j2).abs() - 1e-9 || j3 > j1 + j2 + 1e-9 {
        return Ok(0.0);
    }
    // j1+j2+j3 must be an integer for the symbol to exist at all
    if !is_integral(j1 + j2 + j3) {
        return Ok(0.0);
    }
    // with all m zero, an odd total vanishes by parity
    if m1.abs() < 1e-9 && m2.abs() < 1e-9 && m3.abs() < 1e-9 {
        let t = (j1 + j2 + j3).round() as i64;
        if t % 2 != 0 {
            return Ok(0.0);
        }
    }

    let Some(ln_tri) = ln_delta(j1, j2, j3) else {
        return Ok(0.0);
    };

    // sqrt of the six (j +/- m)! factors
    let mut ln_pre = ln_tri;
    for (j, m) in [(j1, m1), (j2, m2), (j3, m3)] {
        let (Some(p), Some(q)) = (idx(j + m), idx(j - m)) else {
            return Ok(0.0);
        };
        ln_pre += ln_fact(p) + ln_fact(q);
    }
    ln_pre *= 0.5;

    // ---- Racah's alternating sum over k ----------------------------
    // Terms exist only where every factorial argument is >= 0.
    let k_lo = 0.0_f64
        .max(j2 - j3 - m1)
        .max(j1 - j3 + m2)
        .round()
        .max(0.0);
    let k_hi = (j1 + j2 - j3).min(j1 - m1).min(j2 + m2).round();
    if k_hi < k_lo {
        return Ok(0.0);
    }

    let mut sum = 0.0_f64;
    let mut k = k_lo;
    while k <= k_hi + 1e-9 {
        let args = [
            idx(k),
            idx(j1 + j2 - j3 - k),
            idx(j1 - m1 - k),
            idx(j2 + m2 - k),
            idx(j3 - j2 + m1 + k),
            idx(j3 - j1 - m2 + k),
        ];
        if let [Some(a), Some(b), Some(c), Some(d), Some(e), Some(f)] = args {
            let ln_den =
                ln_fact(a) + ln_fact(b) + ln_fact(c) + ln_fact(d) + ln_fact(e) + ln_fact(f);
            let sign = if (k.round() as i64) % 2 == 0 { 1.0 } else { -1.0 };
            sum += sign * (ln_pre - ln_den).exp();
        }
        k += 1.0;
    }

    // overall phase (-1)^(j1 - j2 - m3)
    let ph = (j1 - j2 - m3).round() as i64;
    let sign = if ph.rem_euclid(2) == 0 { 1.0 } else { -1.0 };
    Ok(sign * sum)
}

/// The Clebsch–Gordan coefficient `<j1 m1 j2 m2 | j3 m3>`.
///
/// Expressed through the 3-j symbol, which is the numerically better
/// behaved object:
/// `<j1 m1 j2 m2 | j3 m3> = (-1)^(j1-j2+m3) sqrt(2 j3 + 1) (j1 j2 j3 ; m1 m2 -m3)`.
///
/// # Errors
/// As [`wigner_3j`].
///
/// # Examples
/// ```
/// use special_functions::wigner::clebsch_gordan;
/// // two spin-1/2 into the triplet top state: coefficient is exactly 1
/// let v = clebsch_gordan(0.5, 0.5, 0.5, 0.5, 1.0, 1.0).unwrap();
/// assert!((v - 1.0).abs() < 1e-14);
/// // the singlet/triplet m=0 mixing is 1/sqrt(2)
/// let v = clebsch_gordan(0.5, 0.5, 0.5, -0.5, 1.0, 0.0).unwrap();
/// assert!((v - 1.0 / 2.0_f64.sqrt()).abs() < 1e-14);
/// ```
pub fn clebsch_gordan(
    j1: f64,
    m1: f64,
    j2: f64,
    m2: f64,
    j3: f64,
    m3: f64,
) -> Result<f64, String> {
    let three_j = wigner_3j(j1, j2, j3, m1, m2, -m3)?;
    if three_j == 0.0 {
        return Ok(0.0);
    }
    let ph = (j1 - j2 + m3).round() as i64;
    let sign = if ph.rem_euclid(2) == 0 { 1.0 } else { -1.0 };
    Ok(sign * (2.0 * j3 + 1.0).sqrt() * three_j)
}

/// The Wigner 6-j symbol
/// `{ j1 j2 j3 ; j4 j5 j6 }`.
///
/// Governs the recoupling of three angular momenta. Returns `0.0` when
/// any of the four triangle conditions fails.
///
/// The four triads are `(j1,j2,j3)`, `(j1,j5,j6)`, `(j4,j2,j6)` and
/// `(j4,j5,j3)`.
///
/// # Errors
/// An argument that is neither integer nor half-integer, or negative.
///
/// # Examples
/// ```
/// use special_functions::wigner::wigner_6j;
/// // {1 1 1 ; 1 1 1} = 1/6
/// let v = wigner_6j(1.0, 1.0, 1.0, 1.0, 1.0, 1.0).unwrap();
/// assert!((v - 1.0 / 6.0).abs() < 1e-14);
/// ```
pub fn wigner_6j(
    j1: f64,
    j2: f64,
    j3: f64,
    j4: f64,
    j5: f64,
    j6: f64,
) -> Result<f64, String> {
    const NAME: &str = "wigner_6j";
    for (label, j) in [
        ("j1", j1),
        ("j2", j2),
        ("j3", j3),
        ("j4", j4),
        ("j5", j5),
        ("j6", j6),
    ] {
        check_j(NAME, label, j)?;
    }

    // The four triads; each must satisfy the triangle rule AND sum to
    // an integer.
    let triads = [
        (j1, j2, j3),
        (j1, j5, j6),
        (j4, j2, j6),
        (j4, j5, j3),
    ];
    let mut ln_pre = 0.0_f64;
    for (a, b, c) in triads {
        if !is_integral(a + b + c) {
            return Ok(0.0);
        }
        if c < (a - b).abs() - 1e-9 || c > a + b + 1e-9 {
            return Ok(0.0);
        }
        match ln_delta(a, b, c) {
            Some(d) => ln_pre += d,
            None => return Ok(0.0),
        }
    }
    ln_pre *= 0.5;

    // Racah's sum: k runs over the range where every factorial is >= 0.
    let sums = [
        j1 + j2 + j3,
        j1 + j5 + j6,
        j4 + j2 + j6,
        j4 + j5 + j3,
    ];
    let prods = [
        j1 + j2 + j4 + j5,
        j2 + j3 + j5 + j6,
        j1 + j3 + j4 + j6,
    ];
    let k_lo = sums.iter().cloned().fold(f64::NEG_INFINITY, f64::max).round();
    let k_hi = prods.iter().cloned().fold(f64::INFINITY, f64::min).round();
    if k_hi < k_lo {
        return Ok(0.0);
    }

    let mut sum = 0.0_f64;
    let mut k = k_lo;
    while k <= k_hi + 1e-9 {
        let mut ok = true;
        let mut ln_den = 0.0_f64;
        for s in sums {
            match idx(k - s) {
                Some(v) => ln_den += ln_fact(v),
                None => ok = false,
            }
        }
        for p in prods {
            match idx(p - k) {
                Some(v) => ln_den += ln_fact(v),
                None => ok = false,
            }
        }
        if ok {
            if let Some(kp1) = idx(k + 1.0) {
                let sign = if (k.round() as i64).rem_euclid(2) == 0 { 1.0 } else { -1.0 };
                sum += sign * (ln_fact(kp1) + ln_pre - ln_den).exp();
            }
        }
        k += 1.0;
    }
    Ok(sum)
}

/// The Wigner 9-j symbol
/// `{ j1 j2 j12 ; j3 j4 j34 ; j13 j24 j }`, written row by row.
///
/// Governs the recoupling of **four** angular momenta: it is the
/// overlap between coupling (1,2) and (3,4) first, versus coupling
/// (1,3) and (2,4) first.
///
/// Evaluated as a single sum over 6-j symbols
/// (DLMF 34.6.1, <https://dlmf.nist.gov/34.6.E1>; Varshalovich §10.2):
///
/// ```text
/// {a b c}
/// {d e f} = sum_x (-1)^(2x) (2x+1) {a b c}{d e f}{g h i}
/// {g h i}                          {f i x}{b x h}{x a d}
/// ```
///
/// Returns `0.0` when any of the six triangle conditions fails — the
/// correct value, not an error.
///
/// # Errors
/// An argument that is neither integer nor half-integer, or negative.
///
/// # Examples
/// ```
/// use special_functions::wigner::wigner_9j;
/// // With a zero in the corner the 9-j collapses to a 6-j:
/// //   {a b c; d e f; g h 0} = delta_cf delta_gh (-1)^(b+c+d+g)
/// //                            / sqrt((2c+1)(2g+1)) * {a b c; e d g}
/// let v = wigner_9j(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0).unwrap();
/// assert!(v.is_finite());
/// // a broken triangle vanishes
/// assert_eq!(wigner_9j(1.0, 1.0, 9.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0).unwrap(), 0.0);
/// ```
#[allow(clippy::too_many_arguments)]
pub fn wigner_9j(
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    e: f64,
    f: f64,
    g: f64,
    h: f64,
    i: f64,
) -> Result<f64, String> {
    const NAME: &str = "wigner_9j";
    for (label, j) in [
        ("j1", a), ("j2", b), ("j12", c),
        ("j3", d), ("j4", e), ("j34", f),
        ("j13", g), ("j24", h), ("j", i),
    ] {
        check_j(NAME, label, j)?;
    }

    // All six triads of the array must close, or the symbol is zero.
    for (p, q, r) in [
        (a, b, c), (d, e, f), (g, h, i), // rows
        (a, d, g), (b, e, h), (c, f, i), // columns
    ] {
        if !is_integral(p + q + r) || r < (p - q).abs() - 1e-9 || r > p + q + 1e-9 {
            return Ok(0.0);
        }
    }

    // x runs over the values allowed by all three 6-j symbols at once.
    let lo = (a - i).abs().max((d - h).abs()).max((b - f).abs());
    let hi = (a + i).min(d + h).min(b + f);
    if hi < lo - 1e-9 {
        return Ok(0.0);
    }

    let mut sum = 0.0_f64;
    let mut x = lo;
    while x <= hi + 1e-9 {
        let t1 = wigner_6j(a, b, c, f, i, x)?;
        if t1 != 0.0 {
            let t2 = wigner_6j(d, e, f, b, x, h)?;
            if t2 != 0.0 {
                let t3 = wigner_6j(g, h, i, x, a, d)?;
                if t3 != 0.0 {
                    // (-1)^(2x) is +1 for integer x and -1 for
                    // half-integer x; 2x is always an integer here.
                    let ph = (2.0 * x).round() as i64;
                    let sign = if ph.rem_euclid(2) == 0 { 1.0 } else { -1.0 };
                    sum += sign * (2.0 * x + 1.0) * t1 * t2 * t3;
                }
            }
        }
        x += 1.0;
    }
    Ok(sum)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    /// Exact closed forms, from the standard tables.
    #[test]
    fn known_3j_values() {
        let s3 = 3.0_f64.sqrt();
        // (0 0 0; 0 0 0) = 1
        assert!(close(wigner_3j(0.0, 0.0, 0.0, 0.0, 0.0, 0.0).unwrap(), 1.0, 1e-14));
        // (1 1 0; 0 0 0) = -1/sqrt(3)
        assert!(close(
            wigner_3j(1.0, 1.0, 0.0, 0.0, 0.0, 0.0).unwrap(),
            -1.0 / s3,
            1e-14
        ));
        // (1 1 2; 0 0 0) = sqrt(2/15)
        assert!(close(
            wigner_3j(1.0, 1.0, 2.0, 0.0, 0.0, 0.0).unwrap(),
            (2.0_f64 / 15.0).sqrt(),
            1e-14
        ));
        // (1 1 1; 0 0 0) = 0 by parity (odd sum with all m = 0)
        assert_eq!(wigner_3j(1.0, 1.0, 1.0, 0.0, 0.0, 0.0).unwrap(), 0.0);
        // (1/2 1/2 1; 1/2 -1/2 0) = 1/sqrt(6)
        assert!(close(
            wigner_3j(0.5, 0.5, 1.0, 0.5, -0.5, 0.0).unwrap(),
            1.0 / 6.0_f64.sqrt(),
            1e-14
        ));
        // (2 2 0; 0 0 0) = 1/sqrt(5)
        assert!(close(
            wigner_3j(2.0, 2.0, 0.0, 0.0, 0.0, 0.0).unwrap(),
            1.0 / 5.0_f64.sqrt(),
            1e-14
        ));
    }

    /// The general closed form `(j j 0; m -m 0) = (-1)^(j-m)/sqrt(2j+1)`
    /// checked across a range — a much stronger statement than a few
    /// spot values.
    #[test]
    fn zero_coupling_closed_form_over_a_range() {
        for jj in 0..=8 {
            let j = jj as f64 / 2.0; // covers half-integers too
            let mut m = -j;
            while m <= j + 1e-9 {
                let got = wigner_3j(j, j, 0.0, m, -m, 0.0).unwrap();
                let ph = (j - m).round() as i64;
                let want = if ph.rem_euclid(2) == 0 { 1.0 } else { -1.0 }
                    / (2.0 * j + 1.0).sqrt();
                assert!(close(got, want, 1e-13), "j={j} m={m}: {got} vs {want}");
                m += 1.0;
            }
        }
    }

    /// Permutation symmetry: even permutations of the columns leave the
    /// 3-j alone, odd ones multiply it by (-1)^(j1+j2+j3).
    #[test]
    fn permutation_and_reflection_symmetries() {
        let cases = [
            (2.0, 1.0, 2.0, 1.0, -1.0, 0.0),
            (1.5, 1.5, 2.0, 0.5, 0.5, -1.0),
            (3.0, 2.0, 3.0, -1.0, 2.0, -1.0),
        ];
        for (j1, j2, j3, m1, m2, m3) in cases {
            let base = wigner_3j(j1, j2, j3, m1, m2, m3).unwrap();
            // cyclic (even) permutation
            let cyc = wigner_3j(j2, j3, j1, m2, m3, m1).unwrap();
            assert!(close(base, cyc, 1e-13), "cyclic symmetry");
            // swap columns 1 and 2 (odd)
            let ph = (j1 + j2 + j3).round() as i64;
            let s = if ph.rem_euclid(2) == 0 { 1.0 } else { -1.0 };
            let sw = wigner_3j(j2, j1, j3, m2, m1, m3).unwrap();
            assert!(close(sw, s * base, 1e-13), "column swap");
            // flipping every m carries the same phase
            let fl = wigner_3j(j1, j2, j3, -m1, -m2, -m3).unwrap();
            assert!(close(fl, s * base, 1e-13), "m reflection");
        }
    }

    /// Orthogonality of the Clebsch–Gordan coefficients:
    /// `sum over m1,m2 of <j1 m1 j2 m2|j3 m3>^2 = 1`.
    ///
    /// This is the test that is actually sensitive to cancellation
    /// error in the Racah sum, which is why it is here rather than only
    /// a list of spot values.
    #[test]
    fn clebsch_gordan_rows_are_normalised() {
        for (j1, j2) in [(0.5_f64, 0.5_f64), (1.0, 1.0), (1.5, 1.0), (2.0, 1.5), (3.0, 2.0)] {
            let mut j3 = (j1 - j2).abs();
            while j3 <= j1 + j2 + 1e-9 {
                let mut m3 = -j3;
                while m3 <= j3 + 1e-9 {
                    let mut total = 0.0;
                    let mut m1 = -j1;
                    while m1 <= j1 + 1e-9 {
                        let m2 = m3 - m1;
                        if m2.abs() <= j2 + 1e-9 {
                            let c = clebsch_gordan(j1, m1, j2, m2, j3, m3).unwrap();
                            total += c * c;
                        }
                        m1 += 1.0;
                    }
                    assert!(
                        close(total, 1.0, 1e-12),
                        "j1={j1} j2={j2} j3={j3} m3={m3}: sum was {total}"
                    );
                    m3 += 1.0;
                }
                j3 += 1.0;
            }
        }
    }

    #[test]
    fn known_clebsch_gordan_values() {
        // |1,1> = |up up>
        assert!(close(
            clebsch_gordan(0.5, 0.5, 0.5, 0.5, 1.0, 1.0).unwrap(),
            1.0,
            1e-14
        ));
        // |1,0> = (|up down> + |down up>)/sqrt(2)
        let r2 = 1.0 / 2.0_f64.sqrt();
        assert!(close(
            clebsch_gordan(0.5, 0.5, 0.5, -0.5, 1.0, 0.0).unwrap(),
            r2,
            1e-14
        ));
        // |0,0> = (|up down> - |down up>)/sqrt(2)
        assert!(close(
            clebsch_gordan(0.5, 0.5, 0.5, -0.5, 0.0, 0.0).unwrap(),
            r2,
            1e-14
        ));
        assert!(close(
            clebsch_gordan(0.5, -0.5, 0.5, 0.5, 0.0, 0.0).unwrap(),
            -r2,
            1e-14
        ));
    }

    #[test]
    fn known_6j_values() {
        // {1 1 1; 1 1 1} = 1/6
        assert!(close(
            wigner_6j(1.0, 1.0, 1.0, 1.0, 1.0, 1.0).unwrap(),
            1.0 / 6.0,
            1e-14
        ));
        // {1 1 2; 1 1 2} = +1/30. (An earlier version of this test
        // asserted -1/30 from memory; the orthogonality test below,
        // which fixes sign AND normalisation absolutely, says +1/30.
        // Recalled table values are the weakest evidence here.)
        assert!(close(
            wigner_6j(1.0, 1.0, 2.0, 1.0, 1.0, 2.0).unwrap(),
            1.0 / 30.0,
            1e-13
        ));
    }

    /// 6-j orthogonality:
    /// `sum_x (2x+1) {a b x; c d p} {a b x; c d q} = delta_pq / (2p+1)`.
    ///
    /// The right-hand side is an ABSOLUTE value, so this pins the
    /// normalisation and the sign convention without reference to any
    /// table — which matters, because a table value quoted from memory
    /// was wrong above.
    #[test]
    fn six_j_orthogonality_relation() {
        for (a, b, c, d) in [(1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64), (2.0, 1.0, 2.0, 1.0), (1.5, 1.5, 1.0, 1.0)] {
            // p and q must themselves be coupleable: the symbol
            // {a b x; c d p} contains the triads (a,d,p) and (c,b,p),
            // so p is bounded by BOTH, not merely by a+b. Outside that
            // range every term vanishes by the triangle rule and the
            // relation simply does not apply. (An earlier version of
            // this test started p at 0 and reported a failure at
            // a=2,b=1,c=2,d=1,p=0 — where |a-d| = 1 > 0. The bound was
            // wrong, not the symbol.)
            let p_lo = (a - d).abs().max((c - b).abs());
            let p_hi = (a + d).min(c + b);
            let mut p = p_lo;
            while p <= p_hi + 1e-9 {
                let mut q = p_lo;
                while q <= p_hi + 1e-9 {
                    let mut sum = 0.0;
                    let mut x = 0.0_f64;
                    while x <= a + b + 1e-9 {
                        sum += (2.0 * x + 1.0)
                            * wigner_6j(a, b, x, c, d, p).unwrap()
                            * wigner_6j(a, b, x, c, d, q).unwrap();
                        x += 1.0;
                    }
                    let want = if (p - q).abs() < 1e-9 { 1.0 / (2.0 * p + 1.0) } else { 0.0 };
                    assert!(
                        close(sum, want, 1e-12),
                        "a={a} b={b} c={c} d={d} p={p} q={q}: {sum} vs {want}"
                    );
                    q += 1.0;
                }
                p += 1.0;
            }
        }
    }

    /// The closed form `{a b c; 0 c b} = (-1)^(a+b+c)/sqrt((2b+1)(2c+1))`
    /// swept over a range, which exercises the triangle logic far more
    /// thoroughly than isolated values.
    #[test]
    fn six_j_with_a_zero_argument_closed_form() {
        for bb in 0..=6 {
            for cc in 0..=6 {
                let b = bb as f64 / 2.0;
                let c = cc as f64 / 2.0;
                let mut a = (b - c).abs();
                while a <= b + c + 1e-9 {
                    if !is_integral(a + b + c) {
                        a += 1.0;
                        continue;
                    }
                    let got = wigner_6j(a, b, c, 0.0, c, b).unwrap();
                    let ph = (a + b + c).round() as i64;
                    let want = if ph.rem_euclid(2) == 0 { 1.0 } else { -1.0 }
                        / ((2.0 * b + 1.0) * (2.0 * c + 1.0)).sqrt();
                    assert!(close(got, want, 1e-12), "a={a} b={b} c={c}: {got} vs {want}");
                    a += 1.0;
                }
            }
        }
    }

    /// With a zero in the corner the 9-j collapses to a 6-j:
    ///
    /// ```text
    ///   {a b c; d e f; g h 0} = delta_cf delta_gh (-1)^(b+c+d+g)
    ///                            / sqrt((2c+1)(2g+1)) * {a b c; e d g}
    /// ```
    ///
    /// Swept over a range rather than spot-checked, and it fixes the
    /// overall normalisation and phase absolutely — no table needed.
    #[test]
    fn nine_j_with_a_zero_reduces_to_a_six_j() {
        let mut checked = 0;
        for aa in 0..=4u32 {
            for bb in 0..=4u32 {
                for cc in 0..=4u32 {
                    let (a, b, c) = (aa as f64 / 2.0, bb as f64 / 2.0, cc as f64 / 2.0);
                    // the reduction needs f = c and h = g
                    for dd in 0..=4u32 {
                        for gg in 0..=4u32 {
                            let (d, g) = (dd as f64 / 2.0, gg as f64 / 2.0);
                            let (e, f, h) = (d, c, g);
                            let got = wigner_9j(a, b, c, d, e, f, g, h, 0.0).unwrap();
                            let six = wigner_6j(a, b, c, e, d, g).unwrap();
                            let ph = (b + c + d + g).round() as i64;
                            let sign = if ph.rem_euclid(2) == 0 { 1.0 } else { -1.0 };
                            let want = if is_integral(b + c + d + g) {
                                sign * six
                                    / ((2.0 * c + 1.0) * (2.0 * g + 1.0)).sqrt()
                            } else {
                                // the phase is not defined unless the
                                // exponent is an integer; those cases
                                // have a vanishing 6-j anyway
                                assert!(six.abs() < 1e-12);
                                0.0
                            };
                            assert!(
                                (got - want).abs() < 1e-11,
                                "9j({a},{b},{c};{d},{e},{f};{g},{h},0) = {got}, want {want}"
                            );
                            checked += 1;
                        }
                    }
                }
            }
        }
        assert!(checked > 500, "only {checked} cases exercised");
    }

    /// Orthogonality of the 9-j symbols:
    ///
    /// ```text
    ///  sum_{c,f} (2c+1)(2f+1) {a b c; d e f; g h i} {a b c; d e f; g' h' i}
    ///     = delta_gg' delta_hh' / ((2g+1)(2h+1))
    /// ```
    ///
    /// when (g,h,i) and (g',h',i) both close. The right-hand side is an
    /// ABSOLUTE value, so this pins the normalisation without reference
    /// to any table — which matters, because a table value recalled
    /// from memory has already been wrong twice in this module.
    #[test]
    fn nine_j_orthogonality() {
        let (a, b, d, e, i) = (1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64);
        let tri = |p: f64, q: f64, r: f64| {
            is_integral(p + q + r) && r >= (p - q).abs() - 1e-9 && r <= p + q + 1e-9
        };
        let mut tested = 0;
        for gg in 0..=4u32 {
            for hh in 0..=4u32 {
                for gg2 in 0..=4u32 {
                    for hh2 in 0..=4u32 {
                        let (g, h) = (gg as f64 / 2.0, hh as f64 / 2.0);
                        let (g2, h2) = (gg2 as f64 / 2.0, hh2 as f64 / 2.0);
                        if !tri(g, h, i) || !tri(g2, h2, i) {
                            continue;
                        }
                        if !tri(a, d, g) || !tri(b, e, h) || !tri(a, d, g2) || !tri(b, e, h2) {
                            continue;
                        }
                        let mut sum = 0.0;
                        let mut c = (a - b).abs();
                        while c <= a + b + 1e-9 {
                            let mut f = (d - e).abs();
                            while f <= d + e + 1e-9 {
                                sum += (2.0 * c + 1.0)
                                    * (2.0 * f + 1.0)
                                    * wigner_9j(a, b, c, d, e, f, g, h, i).unwrap()
                                    * wigner_9j(a, b, c, d, e, f, g2, h2, i).unwrap();
                                f += 1.0;
                            }
                            c += 1.0;
                        }
                        let want = if (g - g2).abs() < 1e-9 && (h - h2).abs() < 1e-9 {
                            1.0 / ((2.0 * g + 1.0) * (2.0 * h + 1.0))
                        } else {
                            0.0
                        };
                        assert!(
                            (sum - want).abs() < 1e-10,
                            "g={g} h={h} g'={g2} h'={h2}: sum {sum}, want {want}"
                        );
                        tested += 1;
                    }
                }
            }
        }
        assert!(tested > 10, "only {tested} combinations were reachable");
    }

    /// The 9-j is invariant under transposition, and under an ODD
    /// permutation of rows or columns it picks up `(-1)^S` where `S` is
    /// the sum of all nine arguments.
    #[test]
    fn nine_j_symmetries() {
        let cases = [
            (1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0),
            (0.5, 0.5, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 2.0),
            (1.5, 1.0, 1.5, 1.0, 1.0, 1.0, 0.5, 1.0, 1.5),
        ];
        for (a, b, c, d, e, f, g, h, i) in cases {
            let base = wigner_9j(a, b, c, d, e, f, g, h, i).unwrap();
            // transpose: rows <-> columns
            let tr = wigner_9j(a, d, g, b, e, h, c, f, i).unwrap();
            assert!((tr - base).abs() < 1e-12, "transpose changed it");
            let s = (a + b + c + d + e + f + g + h + i).round() as i64;
            let sign = if s.rem_euclid(2) == 0 { 1.0 } else { -1.0 };
            // swap rows 1 and 2 (odd permutation)
            let rs = wigner_9j(d, e, f, a, b, c, g, h, i).unwrap();
            assert!((rs - sign * base).abs() < 1e-12, "row swap phase wrong");
            // swap columns 1 and 2 (odd permutation)
            let cs = wigner_9j(b, a, c, e, d, f, h, g, i).unwrap();
            assert!((cs - sign * base).abs() < 1e-12, "column swap phase wrong");
            // cyclic row permutation is even: unchanged
            let cyc = wigner_9j(d, e, f, g, h, i, a, b, c).unwrap();
            assert!((cyc - base).abs() < 1e-12, "cyclic row permutation changed it");
        }
    }

    /// Selection-rule violations are ZERO, not errors — that is the
    /// mathematically correct answer and callers rely on it.
    #[test]
    fn selection_rules_return_zero_not_errors() {
        // m does not sum to zero
        assert_eq!(wigner_3j(1.0, 1.0, 2.0, 1.0, 0.0, 0.0).unwrap(), 0.0);
        // |m| > j
        assert_eq!(wigner_3j(1.0, 1.0, 2.0, 2.0, -2.0, 0.0).unwrap(), 0.0);
        // triangle rule broken
        assert_eq!(wigner_3j(1.0, 1.0, 5.0, 0.0, 0.0, 0.0).unwrap(), 0.0);
        // 6j triangle broken
        assert_eq!(wigner_6j(1.0, 1.0, 9.0, 1.0, 1.0, 1.0).unwrap(), 0.0);
        // 9j: any of the six triads failing gives zero
        assert_eq!(wigner_9j(1.0, 1.0, 9.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0).unwrap(), 0.0);
        assert_eq!(wigner_9j(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 9.0, 1.0, 1.0).unwrap(), 0.0);
    }

    /// Arguments that are not angular momenta at all ARE errors.
    #[test]
    fn non_angular_momentum_arguments_are_errors() {
        assert!(wigner_3j(0.3, 1.0, 1.0, 0.0, 0.0, 0.0).is_err(), "j = 0.3");
        assert!(wigner_3j(-1.0, 1.0, 1.0, 0.0, 0.0, 0.0).is_err(), "negative j");
        // integer j with half-integer m is incoherent
        assert!(wigner_3j(1.0, 1.0, 1.0, 0.5, -0.5, 0.0).is_err(), "j/m mismatch");
        assert!(wigner_6j(0.3, 1.0, 1.0, 1.0, 1.0, 1.0).is_err());
        assert!(wigner_9j(0.3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0).is_err());
        assert!(wigner_9j(-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0).is_err());
    }

    /// Large-j sanity: the closed form must still hold where the
    /// factorials would have overflowed f64 if formed directly.
    /// (j1+j2+j3+1)! at j = 60 is 181!, far past the f64 limit of 170!.
    #[test]
    fn large_j_does_not_overflow() {
        let j = 60.0;
        let v = wigner_3j(j, j, 0.0, 10.0, -10.0, 0.0).unwrap();
        let want = 1.0 / (2.0 * j + 1.0).sqrt(); // (j-m) = 50, even
        assert!(v.is_finite(), "overflowed");
        assert!(close(v, want, 1e-12), "{v} vs {want}");
    }
}
