//! Bridge from the posim expression language to the `special_functions`
//! crate.
//!
//! The project rule is lockstep: a function is not "added" until it is
//! callable from the language, listed in `HELP`, described in the EBNF
//! comment in `parser.rs`, and documented in `grammar.md` / `grammar.tex`
//! with the PDF recompiled. This module is the first of those.
//!
//! # Two kinds of argument checking
//!
//! Everything arrives from the VM as `Value::Num(f64)`, but many of
//! these functions take an *integer order*. A silent `as i32` would turn
//! `legendre_p(2.5, x)` into `legendre_p(2, x)` and return a confident
//! wrong answer, so [`as_int`] rejects any non-integral value by name
//! and position. That validation is the real parser-level work these
//! additions require — the call syntax itself already existed.
//!
//! # Complex values
//!
//! `Value::Complex` and the imaginary literal `3i` were added so the
//! complex Crank–Nicolson solvers could be reached from the language.
//! Real entries promote automatically, so a real band with a complex
//! right-hand side works in a single call.

use special_functions as sf;

use crate::vm::Value;

/// Human-readable type name, for error messages.
fn tn(v: &Value) -> &'static str {
    match v {
        Value::Num(_) => "number",
        Value::Complex(_) => "complex number",
        Value::Vec3(_) => "vector",
        Value::Quat(_) => "quaternion",
        Value::Mat3(_) => "matrix",
        Value::List(_) => "list",
        Value::Str(_) => "string",
        Value::Unit => "nothing",
    }
}

fn as_num(name: &str, pos: usize, v: &Value) -> Result<f64, String> {
    match v {
        Value::Num(n) => Ok(*n),
        other => Err(format!(
            "{name}(): argument {} must be a number, got {}",
            pos + 1,
            tn(other)
        )),
    }
}

/// An integer-valued argument. Rejects a non-integral number outright
/// rather than truncating: `hermite_h(2.5, x)` is a mistake, and
/// silently answering `hermite_h(2, x)` would hide it.
fn as_int(name: &str, pos: usize, v: &Value) -> Result<i32, String> {
    let x = as_num(name, pos, v)?;
    if !x.is_finite() || x.fract() != 0.0 {
        return Err(format!(
            "{name}(): argument {} must be a whole number (an integer order), got {x}",
            pos + 1
        ));
    }
    if !(i32::MIN as f64..=i32::MAX as f64).contains(&x) {
        return Err(format!("{name}(): argument {} is out of range: {x}", pos + 1));
    }
    Ok(x as i32)
}

fn as_usize(name: &str, pos: usize, v: &Value) -> Result<usize, String> {
    let n = as_int(name, pos, v)?;
    if n < 0 {
        return Err(format!(
            "{name}(): argument {} must be zero or positive, got {n}",
            pos + 1
        ));
    }
    Ok(n as usize)
}

/// A flat list of numbers.
///
/// The bracket literal `[...]` is OVERLOADED in this language: three
/// entries make a vector, four make a quaternion, and any other count
/// makes a list. That is fine for physics and surprising here, so all
/// three shapes are accepted wherever a numeric list is wanted —
/// otherwise `solve_tridiag([0,1,1], ...)` would be told its vector is
/// not a list, and a 4x4 matrix row would be called a quaternion. Both
/// are true and useless.
///
/// The quaternion is unpacked w-first, which is the order the user
/// typed: `[a, b, c, d]` parses to `w=a, x=b, y=c, z=d`.
fn as_num_list(name: &str, pos: usize, v: &Value) -> Result<Vec<f64>, String> {
    match v {
        Value::Vec3(u) => Ok(vec![u.x, u.y, u.z]),
        Value::Quat(q) => Ok(vec![q.w, q.x, q.y, q.z]),
        Value::List(items) => items
            .iter()
            .enumerate()
            .map(|(i, it)| match it {
                Value::Num(n) => Ok(*n),
                other => Err(format!(
                    "{name}(): argument {} element {i} must be a number, got {}",
                    pos + 1,
                    tn(other)
                )),
            })
            .collect(),
        other => Err(format!(
            "{name}(): argument {} must be a list of numbers, got {}",
            pos + 1,
            tn(other)
        )),
    }
}

/// A list of equal-length lists: a dense square matrix.
fn as_matrix(name: &str, pos: usize, v: &Value) -> Result<Vec<Vec<f64>>, String> {
    // A 3x3 `Mat3` is the language's native matrix, so accept it
    // directly rather than making the user unpack it into rows.
    if let Value::Mat3(m) = v {
        return Ok(m.0.iter().map(|r| r.to_vec()).collect());
    }
    let rows = match v {
        Value::List(items) => items,
        other => {
            return Err(format!(
                "{name}(): argument {} must be a list of rows, got {}",
                pos + 1,
                tn(other)
            ))
        }
    };
    if rows.is_empty() {
        return Err(format!("{name}(): argument {} is an empty matrix", pos + 1));
    }
    let m: Vec<Vec<f64>> = rows
        .iter()
        .map(|r| as_num_list(name, pos, r))
        .collect::<Result<_, _>>()?;
    let n = m.len();
    if let Some(bad) = m.iter().position(|r| r.len() != n) {
        return Err(format!(
            "{name}(): matrix must be square — it has {n} rows but row {bad} has {} entries",
            m[bad].len()
        ));
    }
    Ok(m)
}

/// A list of complex numbers. Real entries promote, so a user can write
/// a real band and a complex right-hand side in the same call.
fn as_cplx_list(name: &str, pos: usize, v: &Value) -> Result<Vec<sf::complex::Complex64>, String> {
    let items: Vec<Value> = match v {
        Value::List(items) => items.clone(),
        Value::Vec3(u) => vec![Value::Num(u.x), Value::Num(u.y), Value::Num(u.z)],
        Value::Quat(q) => vec![
            Value::Num(q.w),
            Value::Num(q.x),
            Value::Num(q.y),
            Value::Num(q.z),
        ],
        other => {
            return Err(format!(
                "{name}(): argument {} must be a list of numbers, got {}",
                pos + 1,
                tn(other)
            ))
        }
    };
    items
        .iter()
        .enumerate()
        .map(|(i, it)| match it {
            Value::Num(x) => Ok(sf::complex::Complex64::real(*x)),
            Value::Complex(z) => Ok(*z),
            other => Err(format!(
                "{name}(): argument {} element {i} must be a number, got {}",
                pos + 1,
                tn(other)
            )),
        })
        .collect()
}

fn cplx(v: Vec<sf::complex::Complex64>) -> Value {
    Value::List(v.into_iter().map(Value::Complex).collect())
}

fn as_cplx(name: &str, pos: usize, v: &Value) -> Result<sf::complex::Complex64, String> {
    match v {
        Value::Num(x) => Ok(sf::complex::Complex64::real(*x)),
        Value::Complex(z) => Ok(*z),
        other => Err(format!(
            "{name}(): argument {} must be a number, got {}",
            pos + 1,
            tn(other)
        )),
    }
}

fn nums(v: Vec<f64>) -> Value {
    Value::List(v.into_iter().map(Value::Num).collect())
}

/// Dispatch a special-function call.
///
/// Returns `None` if `name` is not one of ours, so the caller can fall
/// through to the core builtins and then to its own "unknown function"
/// error — this module never swallows a name it does not own.
pub fn call(name: &str, args: &[Value]) -> Option<Result<Value, String>> {
    // Arity is checked once, here, so each arm below can index freely.
    let want: usize = match name {
        "rel_err" => 2,
        "sph_j" | "sph_y" | "sph_j_prime" | "sph_y_prime" | "legendre_p" | "legendre_p_prime"
        | "hermite_h" | "hermite_he" | "laguerre_l" | "chebyshev_t" | "chebyshev_u"
        | "bessel_j" | "bessel_j_array" => 2,
        "assoc_legendre_p" | "norm_assoc_legendre_p" | "laguerre_l_assoc" | "gegenbauer_c" => 3,
        "sph_harm" | "sph_harm_real" | "jacobi_p" => 4,
        "gauss_legendre" => 1,
        // Angular momenta may be HALF-integers, so these take plain
        // numbers and validate in the library rather than here.
        "wigner_3j" | "wigner_6j" | "clebsch_gordan" => 6,
        "wigner_9j" => 9,
        "eigenvalues" | "jacobi_eigen" => 1,
        "solve_tridiag" | "solve_tridiag_c" => 4,
        // Complex argument: these take and return complex values, which
        // is why the language needed Value::Complex before they could
        // exist at all.
        "bessel_j_z" | "bessel_i_z" | "bessel_y_z" | "bessel_k_z" => 2,
        "bessel_j_nu" | "bessel_i_nu" | "bessel_y_nu" | "bessel_k_nu" => 2,
        "gamma_z" | "ln_gamma_z" | "rgamma_z" => 1,
        "airy_z" => 1,
        "hankel_h1_z" | "hankel_h2_z" | "hankel_h1_nu" | "hankel_h2_nu" => 2,
        "hankel_h1_prime_z" | "hankel_h2_prime_z" => 2,
        "hankel_h1_prime_nu" | "hankel_h2_prime_nu" => 2,
        "sph_hankel_h1" | "sph_hankel_h2" => 2,
        "sph_hankel_h1_prime" | "sph_hankel_h2_prime" => 2,
        "bessel_j_scaled" | "bessel_y_scaled" | "bessel_i_scaled" | "bessel_k_scaled" => 2,
        "hankel_h1_scaled" | "hankel_h2_scaled" => 2,
        "solve_cyclic_tridiag_c" => 6,
        _ => return None,
    };
    if args.len() != want {
        return Some(Err(format!(
            "{name}() takes {want} argument(s), got {}",
            args.len()
        )));
    }
    Some(dispatch(name, args))
}

fn dispatch(name: &str, a: &[Value]) -> Result<Value, String> {
    // Two common shapes, to keep the arms one line each.
    let nx = |f: fn(i32, f64) -> Result<f64, String>| -> Result<Value, String> {
        Ok(Value::Num(f(as_int(name, 0, &a[0])?, as_num(name, 1, &a[1])?)?))
    };
    let lmx = |f: fn(i32, i32, f64) -> Result<f64, String>| -> Result<Value, String> {
        Ok(Value::Num(f(
            as_int(name, 0, &a[0])?,
            as_int(name, 1, &a[1])?,
            as_num(name, 2, &a[2])?,
        )?))
    };

    match name {
        // ---- spherical Bessel -------------------------------------
        "sph_j" => nx(sf::sph_bessel::sph_j),
        "sph_y" => nx(sf::sph_bessel::sph_y),
        "sph_j_prime" => nx(sf::sph_bessel::sph_j_prime),
        "sph_y_prime" => nx(sf::sph_bessel::sph_y_prime),

        // ---- Legendre and spherical harmonics ---------------------
        "legendre_p" => nx(sf::legendre::legendre_p),
        "legendre_p_prime" => nx(sf::legendre::legendre_p_prime),
        "assoc_legendre_p" => lmx(sf::legendre::assoc_legendre_p),
        "norm_assoc_legendre_p" => lmx(sf::legendre::norm_assoc_legendre_p),
        "sph_harm_real" => Ok(Value::Num(sf::legendre::sph_harm_real(
            as_int(name, 0, &a[0])?,
            as_int(name, 1, &a[1])?,
            as_num(name, 2, &a[2])?,
            as_num(name, 3, &a[3])?,
        )?)),
        // Complex-valued: returned as the two-element list [re, im],
        // since the language has no complex type yet.
        "sph_harm" => {
            let (re, im) = sf::legendre::sph_harm(
                as_int(name, 0, &a[0])?,
                as_int(name, 1, &a[1])?,
                as_num(name, 2, &a[2])?,
                as_num(name, 3, &a[3])?,
            )?;
            Ok(nums(vec![re, im]))
        }

        // ---- classical orthogonal polynomials ---------------------
        "hermite_h" => nx(sf::orthopoly::hermite_h),
        "hermite_he" => nx(sf::orthopoly::hermite_he),
        "laguerre_l" => nx(sf::orthopoly::laguerre_l),
        "chebyshev_t" => nx(sf::orthopoly::chebyshev_t),
        "chebyshev_u" => nx(sf::orthopoly::chebyshev_u),
        "laguerre_l_assoc" => Ok(Value::Num(sf::orthopoly::laguerre_l_assoc(
            as_int(name, 0, &a[0])?,
            as_num(name, 1, &a[1])?,
            as_num(name, 2, &a[2])?,
        )?)),
        "gegenbauer_c" => Ok(Value::Num(sf::orthopoly::gegenbauer_c(
            as_int(name, 0, &a[0])?,
            as_num(name, 1, &a[1])?,
            as_num(name, 2, &a[2])?,
        )?)),
        "jacobi_p" => Ok(Value::Num(sf::orthopoly::jacobi_p(
            as_int(name, 0, &a[0])?,
            as_num(name, 1, &a[1])?,
            as_num(name, 2, &a[2])?,
            as_num(name, 3, &a[3])?,
        )?)),

        // ---- cylindrical Bessel, integer order --------------------
        "bessel_j" => nx(sf::bessel::bessel_j),
        "bessel_j_array" => Ok(nums(sf::bessel::bessel_j_array(
            as_usize(name, 0, &a[0])?,
            as_num(name, 1, &a[1])?,
        )?)),

        // ---- Bessel, complex argument -----------------------------
        "bessel_j_z" => Ok(Value::Complex(sf::bessel_complex::bessel_j_c(
            as_int(name, 0, &a[0])?,
            as_cplx(name, 1, &a[1])?,
        )?)),
        "bessel_i_z" => Ok(Value::Complex(sf::bessel_complex::bessel_i_c(
            as_int(name, 0, &a[0])?,
            as_cplx(name, 1, &a[1])?,
        )?)),
        "bessel_y_z" => Ok(Value::Complex(sf::bessel_complex::bessel_y_c(
            as_int(name, 0, &a[0])?,
            as_cplx(name, 1, &a[1])?,
        )?)),
        "bessel_k_z" => Ok(Value::Complex(sf::bessel_complex::bessel_k_c(
            as_int(name, 0, &a[0])?,
            as_cplx(name, 1, &a[1])?,
        )?)),

        // ---- Bessel, real non-integer order, complex argument -----
        // Order is a plain number here, not an integer: that is the
        // whole point of these four.
        // The order is taken as COMPLEX here. A real one dispatches to
        // exactly the routine it always did, so nothing that worked
        // before changes; a complex one reaches the series that needs
        // the complex gamma.
        "bessel_j_nu" => Ok(Value::Complex(sf::bessel_cnu::bessel_j_cnu(
            as_cplx(name, 0, &a[0])?,
            as_cplx(name, 1, &a[1])?,
        )?)),
        "bessel_i_nu" => Ok(Value::Complex(sf::bessel_cnu::bessel_i_cnu(
            as_cplx(name, 0, &a[0])?,
            as_cplx(name, 1, &a[1])?,
        )?)),
        "bessel_y_nu" => Ok(Value::Complex(sf::bessel_cnu::bessel_y_cnu(
            as_cplx(name, 0, &a[0])?,
            as_cplx(name, 1, &a[1])?,
        )?)),
        "bessel_k_nu" => Ok(Value::Complex(sf::bessel_cnu::bessel_k_cnu(
            as_cplx(name, 0, &a[0])?,
            as_cplx(name, 1, &a[1])?,
        )?)),

        // ---- Airy at complex argument -----------------------------
        // Returns all four at once — [Ai, Ai', Bi, Bi'] — because the
        // routine computes them together and a caller who wants a
        // Wronskian or a boundary condition wants all four.
        "airy_z" => {
            let v = sf::airy_complex::airy_c(as_cplx(name, 0, &a[0])?)?;
            Ok(Value::List(vec![
                Value::Complex(v.ai),
                Value::Complex(v.aip),
                Value::Complex(v.bi),
                Value::Complex(v.bip),
            ]))
        }

        // ---- gamma at complex argument ----------------------------
        "gamma_z" => Ok(Value::Complex(sf::gamma_complex::gamma_c(as_cplx(
            name, 0, &a[0],
        )?)?)),
        "ln_gamma_z" => Ok(Value::Complex(sf::gamma_complex::ln_gamma_c(as_cplx(
            name, 0, &a[0],
        )?)?)),
        "rgamma_z" => Ok(Value::Complex(sf::gamma_complex::rgamma_c(as_cplx(
            name, 0, &a[0],
        )?)?)),

        // ---- Hankel: the travelling-wave pair ---------------------
        // H1 = J + iY is outgoing, H2 = J - iY incoming. Named rather
        // than left to the user to assemble, because the assembly
        // cancels badly in half the plane and a caller doing it by hand
        // has no way to know — see grammar.md.
        "hankel_h1_z" => Ok(Value::Complex(sf::hankel::hankel_h1_c(
            as_int(name, 0, &a[0])?,
            as_cplx(name, 1, &a[1])?,
        )?)),
        "hankel_h2_z" => Ok(Value::Complex(sf::hankel::hankel_h2_c(
            as_int(name, 0, &a[0])?,
            as_cplx(name, 1, &a[1])?,
        )?)),
        "hankel_h1_nu" => Ok(Value::Complex(sf::hankel::hankel_h1_nu(
            as_num(name, 0, &a[0])?,
            as_cplx(name, 1, &a[1])?,
        )?)),
        "hankel_h2_nu" => Ok(Value::Complex(sf::hankel::hankel_h2_nu(
            as_num(name, 0, &a[0])?,
            as_cplx(name, 1, &a[1])?,
        )?)),
        "hankel_h1_prime_z" => Ok(Value::Complex(sf::hankel::hankel_h1_prime_c(
            as_int(name, 0, &a[0])?,
            as_cplx(name, 1, &a[1])?,
        )?)),
        "hankel_h2_prime_z" => Ok(Value::Complex(sf::hankel::hankel_h2_prime_c(
            as_int(name, 0, &a[0])?,
            as_cplx(name, 1, &a[1])?,
        )?)),
        "hankel_h1_prime_nu" => Ok(Value::Complex(sf::hankel::hankel_h1_prime_nu(
            as_num(name, 0, &a[0])?,
            as_cplx(name, 1, &a[1])?,
        )?)),
        "hankel_h2_prime_nu" => Ok(Value::Complex(sf::hankel::hankel_h2_prime_nu(
            as_num(name, 0, &a[0])?,
            as_cplx(name, 1, &a[1])?,
        )?)),

        // ---- spherical Hankel: REAL argument, complex result ------
        "sph_hankel_h1" => Ok(Value::Complex(sf::hankel::sph_hankel_h1(
            as_int(name, 0, &a[0])?,
            as_num(name, 1, &a[1])?,
        )?)),
        "sph_hankel_h2" => Ok(Value::Complex(sf::hankel::sph_hankel_h2(
            as_int(name, 0, &a[0])?,
            as_num(name, 1, &a[1])?,
        )?)),
        "sph_hankel_h1_prime" => Ok(Value::Complex(sf::hankel::sph_hankel_h1_prime(
            as_int(name, 0, &a[0])?,
            as_num(name, 1, &a[1])?,
        )?)),
        "sph_hankel_h2_prime" => Ok(Value::Complex(sf::hankel::sph_hankel_h2_prime(
            as_int(name, 0, &a[0])?,
            as_num(name, 1, &a[1])?,
        )?)),

        // ---- scaled forms: the exponential factored out -----------
        // These take a real order and return the function DIVIDED by
        // its exponential envelope, computed by a method that never
        // forms the envelope. That is what makes them accurate where
        // the plain forms are not — and defined where the plain forms
        // overflow or underflow out of f64 entirely.
        "bessel_j_scaled" => Ok(Value::Complex(sf::bessel_scaled::bessel_j_scaled_nu(
            as_num(name, 0, &a[0])?,
            as_cplx(name, 1, &a[1])?,
        )?)),
        "bessel_y_scaled" => Ok(Value::Complex(sf::bessel_scaled::bessel_y_scaled_nu(
            as_num(name, 0, &a[0])?,
            as_cplx(name, 1, &a[1])?,
        )?)),
        "bessel_i_scaled" => Ok(Value::Complex(sf::bessel_scaled::bessel_i_scaled_nu(
            as_num(name, 0, &a[0])?,
            as_cplx(name, 1, &a[1])?,
        )?)),
        "bessel_k_scaled" => Ok(Value::Complex(sf::bessel_scaled::bessel_k_scaled_nu(
            as_num(name, 0, &a[0])?,
            as_cplx(name, 1, &a[1])?,
        )?)),
        "hankel_h1_scaled" => Ok(Value::Complex(sf::bessel_scaled::hankel_h1_scaled_nu(
            as_num(name, 0, &a[0])?,
            as_cplx(name, 1, &a[1])?,
        )?)),
        "hankel_h2_scaled" => Ok(Value::Complex(sf::bessel_scaled::hankel_h2_scaled_nu(
            as_num(name, 0, &a[0])?,
            as_cplx(name, 1, &a[1])?,
        )?)),

        // ---- quadrature nodes -------------------------------------
        // Returns [nodes, weights] — two lists, so a script can zip them.
        "gauss_legendre" => {
            let (x, w) = sf::quadrature::gauss_legendre(as_usize(name, 0, &a[0])?)?;
            Ok(Value::List(vec![nums(x), nums(w)]))
        }

        // ---- eigenproblems ----------------------------------------
        "eigenvalues" => Ok(nums(sf::eigen::eigenvalues(&as_matrix(name, 0, &a[0])?)?)),
        // Returns [values, vectors] where vectors is a list of rows.
        "jacobi_eigen" => {
            let (vals, vecs) = sf::eigen::jacobi_eigen(&as_matrix(name, 0, &a[0])?)?;
            Ok(Value::List(vec![
                nums(vals),
                Value::List(vecs.into_iter().map(nums).collect()),
            ]))
        }

        // ---- linear algebra ---------------------------------------
        "solve_tridiag" => Ok(nums(sf::tridiag::solve_tridiag(
            &as_num_list(name, 0, &a[0])?,
            &as_num_list(name, 1, &a[1])?,
            &as_num_list(name, 2, &a[2])?,
            &as_num_list(name, 3, &a[3])?,
        )?)),

        "solve_tridiag_c" => Ok(cplx(sf::tridiag::solve_tridiag_c(
            &as_cplx_list(name, 0, &a[0])?,
            &as_cplx_list(name, 1, &a[1])?,
            &as_cplx_list(name, 2, &a[2])?,
            &as_cplx_list(name, 3, &a[3])?,
        )?)),
        "solve_cyclic_tridiag_c" => Ok(cplx(sf::tridiag::solve_cyclic_tridiag_c(
            &as_cplx_list(name, 0, &a[0])?,
            &as_cplx_list(name, 1, &a[1])?,
            &as_cplx_list(name, 2, &a[2])?,
            as_cplx(name, 4, &a[4])?,
            as_cplx(name, 5, &a[5])?,
            &as_cplx_list(name, 3, &a[3])?,
        )?)),

        // ---- angular-momentum coupling ----------------------------
        "wigner_3j" => Ok(Value::Num(sf::wigner::wigner_3j(
            as_num(name, 0, &a[0])?,
            as_num(name, 1, &a[1])?,
            as_num(name, 2, &a[2])?,
            as_num(name, 3, &a[3])?,
            as_num(name, 4, &a[4])?,
            as_num(name, 5, &a[5])?,
        )?)),
        "wigner_6j" => Ok(Value::Num(sf::wigner::wigner_6j(
            as_num(name, 0, &a[0])?,
            as_num(name, 1, &a[1])?,
            as_num(name, 2, &a[2])?,
            as_num(name, 3, &a[3])?,
            as_num(name, 4, &a[4])?,
            as_num(name, 5, &a[5])?,
        )?)),
        "wigner_9j" => Ok(Value::Num(sf::wigner::wigner_9j(
            as_num(name, 0, &a[0])?,
            as_num(name, 1, &a[1])?,
            as_num(name, 2, &a[2])?,
            as_num(name, 3, &a[3])?,
            as_num(name, 4, &a[4])?,
            as_num(name, 5, &a[5])?,
            as_num(name, 6, &a[6])?,
            as_num(name, 7, &a[7])?,
            as_num(name, 8, &a[8])?,
        )?)),
        "clebsch_gordan" => Ok(Value::Num(sf::wigner::clebsch_gordan(
            as_num(name, 0, &a[0])?,
            as_num(name, 1, &a[1])?,
            as_num(name, 2, &a[2])?,
            as_num(name, 3, &a[3])?,
            as_num(name, 4, &a[4])?,
            as_num(name, 5, &a[5])?,
        )?)),

        // ---- utility ----------------------------------------------
        "rel_err" => Ok(Value::Num(sf::rel_err(
            as_num(name, 0, &a[0])?,
            as_num(name, 1, &a[1])?,
        ))),

        _ => unreachable!("call() already filtered the name set"),
    }
}

/// Every name this module answers to. `vm.rs` folds this into the
/// reserved-name list so a user function cannot shadow one.
pub const SPECIAL_NAMES: &[&str] = &[
    "airy_z",
    "assoc_legendre_p",
    "bessel_i_nu",
    "bessel_i_scaled",
    "bessel_i_z",
    "bessel_j",
    "bessel_k_nu",
    "bessel_k_scaled",
    "bessel_k_z",
    "bessel_j_array",
    "bessel_j_nu",
    "bessel_j_scaled",
    "bessel_j_z",
    "bessel_y_nu",
    "bessel_y_scaled",
    "bessel_y_z",
    "chebyshev_t",
    "chebyshev_u",
    "clebsch_gordan",
    "eigenvalues",
    "gamma_z",
    "gauss_legendre",
    "ln_gamma_z",
    "rgamma_z",
    "hankel_h1_nu",
    "hankel_h1_scaled",
    "hankel_h1_prime_nu",
    "hankel_h1_prime_z",
    "hankel_h1_z",
    "hankel_h2_nu",
    "hankel_h2_scaled",
    "hankel_h2_prime_nu",
    "hankel_h2_prime_z",
    "hankel_h2_z",
    "gegenbauer_c",
    "hermite_h",
    "hermite_he",
    "jacobi_eigen",
    "jacobi_p",
    "laguerre_l",
    "laguerre_l_assoc",
    "legendre_p",
    "legendre_p_prime",
    "norm_assoc_legendre_p",
    "rel_err",
    "solve_cyclic_tridiag_c",
    "solve_tridiag",
    "solve_tridiag_c",
    "sph_hankel_h1",
    "sph_hankel_h1_prime",
    "sph_hankel_h2",
    "sph_hankel_h2_prime",
    "sph_harm",
    "sph_harm_real",
    "sph_j",
    "sph_j_prime",
    "sph_y",
    "sph_y_prime",
    "wigner_3j",
    "wigner_6j",
    "wigner_9j",
];

#[cfg(test)]
mod tests {
    use super::*;

    fn n(x: f64) -> Value {
        Value::Num(x)
    }
    fn call_ok(name: &str, args: &[Value]) -> Value {
        call(name, args).expect("name should be ours").expect("should succeed")
    }
    fn call_err(name: &str, args: &[Value]) -> String {
        call(name, args).expect("name should be ours").unwrap_err()
    }
    fn as_f(v: Value) -> f64 {
        match v {
            Value::Num(x) => x,
            other => panic!("expected a number, got {other:?}"),
        }
    }

    /// Every registered name must dispatch. This is the test that would
    /// have caught the whole gap: a function in the library but absent
    /// from the language.
    #[test]
    fn every_registered_name_is_reachable() {
        for nm in SPECIAL_NAMES {
            assert!(
                call(nm, &[]).is_some(),
                "`{nm}` is in SPECIAL_NAMES but call() does not own it"
            );
            // With zero args it must be an ARITY error, not "unknown".
            let e = call_err(nm, &[]);
            assert!(
                e.contains("takes") && e.contains("argument"),
                "`{nm}` gave an unexpected error: {e}"
            );
        }
    }

    /// The project rule is that a function is not "added" until it is
    /// callable, in HELP, in the parser's EBNF comment, and in BOTH
    /// grammar documents. That rule has been enforced by discipline,
    /// which is to say not enforced. This enforces it: if you register
    /// a function and forget a document, the build fails here.
    #[test]
    fn every_special_function_is_documented_in_lockstep() {
        // `\_` in LaTeX, `\` nowhere else — strip backslashes so one
        // needle works against all four texts.
        let strip = |s: &str| s.replace('\\', "");
        let help = strip(crate::vm::HELP_TEXT);
        let ebnf = strip(include_str!("parser.rs"));
        let md = strip(include_str!("../../grammar.md"));
        let tex = strip(include_str!("../../grammar.tex"));

        let mut missing = Vec::new();
        for nm in SPECIAL_NAMES {
            for (what, hay) in
                [("HELP_TEXT", &help), ("parser.rs EBNF", &ebnf), ("grammar.md", &md), ("grammar.tex", &tex)]
            {
                if !hay.contains(nm) {
                    missing.push(format!("`{nm}` is missing from {what}"));
                }
            }
        }
        assert!(
            missing.is_empty(),
            "grammar lockstep is broken:\n  {}",
            missing.join("\n  ")
        );
    }

    /// Non-integer order, reachable from the language. Checked against
    /// the half-integer closed forms, which are elementary and hold for
    /// complex `z`, plus the two Wronskians.
    #[test]
    fn non_integer_order_bessel_is_reachable() {
        use sf::complex::Complex64 as Cx;
        let z = |re: f64, im: f64| Value::Complex(Cx::new(re, im));
        let get = |v: Value| match v {
            Value::Complex(c) => c,
            Value::Num(r) => Cx::real(r),
            other => panic!("expected complex, got {other:?}"),
        };
        // J_{1/2}(z) = sqrt(2/(pi z)) sin z, for complex z.
        let zz = Cx::new(1.4, 0.7);
        let got = get(call_ok("bessel_j_nu", &[n(0.5), z(zz.re, zz.im)]));
        let csin = ((Cx::I * zz).exp() - (Cx::I * zz * -1.0).exp()) / (Cx::I * 2.0);
        let want = (Cx::real(2.0 / std::f64::consts::PI) * zz.inv()).powf(0.5) * csin;
        assert!((got - want).abs() < 1e-12, "J_1/2: {got:?} vs {want:?}");

        // K_{1/2}(z) = sqrt(pi/(2z)) exp(-z).
        let got = get(call_ok("bessel_k_nu", &[n(0.5), z(zz.re, zz.im)]));
        let want =
            (Cx::real(std::f64::consts::PI * 0.5) * zz.inv()).powf(0.5) * (zz * -1.0).exp();
        assert!((got - want).abs() < 1e-12, "K_1/2: {got:?} vs {want:?}");

        // J-Y Wronskian at a genuinely non-integer order.
        let nu = 1.3;
        let jn = get(call_ok("bessel_j_nu", &[n(nu), z(zz.re, zz.im)]));
        let j1 = get(call_ok("bessel_j_nu", &[n(nu + 1.0), z(zz.re, zz.im)]));
        let yn = get(call_ok("bessel_y_nu", &[n(nu), z(zz.re, zz.im)]));
        let y1 = get(call_ok("bessel_y_nu", &[n(nu + 1.0), z(zz.re, zz.im)]));
        let w = j1 * yn - jn * y1;
        let want = zz.inv() * (2.0 / std::f64::consts::PI);
        assert!((w - want).abs() < 1e-12, "J-Y Wronskian: {w:?} vs {want:?}");

        // I-K Wronskian.
        let iv = get(call_ok("bessel_i_nu", &[n(nu), z(zz.re, zz.im)]));
        let i1 = get(call_ok("bessel_i_nu", &[n(nu + 1.0), z(zz.re, zz.im)]));
        let kv = get(call_ok("bessel_k_nu", &[n(nu), z(zz.re, zz.im)]));
        let k1 = get(call_ok("bessel_k_nu", &[n(nu + 1.0), z(zz.re, zz.im)]));
        let w = iv * k1 + i1 * kv;
        assert!((w - zz.inv()).abs() < 1e-12, "I-K Wronskian: {w:?}");

        // A whole order is accepted here — unlike the _z forms, which
        // reject a fractional one — and agrees with the _z routine.
        let a = get(call_ok("bessel_y_nu", &[n(2.0), z(zz.re, zz.im)]));
        let b = get(call_ok("bessel_y_z", &[n(2.0), z(zz.re, zz.im)]));
        assert!((a - b).abs() < 1e-12, "whole order: {a:?} vs {b:?}");
        assert!(call_err("bessel_j_z", &[n(1.5), z(1.0, 1.0)]).contains("whole number"));
        // ... and the fractional order that the _z form refuses is
        // simply evaluated here (call_ok panics if it errors).
        let _ = call_ok("bessel_j_nu", &[n(1.5), z(1.0, 1.0)]);

        // Singular points still report errors.
        assert!(!call_err("bessel_y_nu", &[n(0.5), z(0.0, 0.0)]).is_empty());
        assert!(!call_err("bessel_k_nu", &[n(0.5), z(0.0, 0.0)]).is_empty());
    }

    /// The Hankel entry points, checked by the properties that make
    /// them worth having as entry points at all.
    #[test]
    fn hankel_entry_points_are_reachable() {
        use sf::complex::Complex64 as Cx;
        let z = |re: f64, im: f64| Value::Complex(Cx::new(re, im));
        let get = |v: Value| match v {
            Value::Complex(c) => c,
            Value::Num(r) => Cx::real(r),
            other => panic!("expected complex, got {other:?}"),
        };

        // H1 = J + iY and H2 = J - iY, checked against the pieces.
        let zz = Cx::new(2.0, -0.6);
        let j = get(call_ok("bessel_j_z", &[n(1.0), z(zz.re, zz.im)]));
        let y = get(call_ok("bessel_y_z", &[n(1.0), z(zz.re, zz.im)]));
        let h1 = get(call_ok("hankel_h1_z", &[n(1.0), z(zz.re, zz.im)]));
        let h2 = get(call_ok("hankel_h2_z", &[n(1.0), z(zz.re, zz.im)]));
        assert!((h1 - (j + Cx::I * y)).abs() < 1e-14, "H1 != J + iY");
        assert!((h2 - (j - Cx::I * y)).abs() < 1e-14, "H2 != J - iY");

        // H1_{1/2}(z) = -i sqrt(2/(pi z)) exp(iz), exactly, and the _nu
        // form is the only one that can be asked for it.
        let got = get(call_ok("hankel_h1_nu", &[n(0.5), z(zz.re, zz.im)]));
        let want = Cx::I * -1.0
            * (Cx::real(2.0 / std::f64::consts::PI) * zz.inv()).powf(0.5)
            * (Cx::I * zz).exp();
        assert!((got - want).abs() < 1e-12, "H1_1/2: {got:?} vs {want:?}");

        // The Hankel Wronskian H1 H2' - H1' H2 = -4i/(pi z).
        let h1p = get(call_ok("hankel_h1_prime_nu", &[n(0.5), z(zz.re, zz.im)]));
        let h2v = get(call_ok("hankel_h2_nu", &[n(0.5), z(zz.re, zz.im)]));
        let h2p = get(call_ok("hankel_h2_prime_nu", &[n(0.5), z(zz.re, zz.im)]));
        let w = got * h2p - h1p * h2v;
        let want = Cx::I * -4.0 * zz.inv() * (1.0 / std::f64::consts::PI);
        assert!((w - want).abs() < 1e-11, "Wronskian: {w:?} vs {want:?}");

        // H1'_0 = -H1_1, which the derivative routine special-cases.
        let a = get(call_ok("hankel_h1_prime_z", &[n(0.0), n(3.0)]));
        let b = get(call_ok("hankel_h1_z", &[n(1.0), n(3.0)]));
        assert!((a + b).abs() < 1e-14, "H1'_0 != -H1_1");
        let a = get(call_ok("hankel_h2_prime_z", &[n(0.0), n(3.0)]));
        let b = get(call_ok("hankel_h2_z", &[n(1.0), n(3.0)]));
        assert!((a + b).abs() < 1e-14, "H2'_0 != -H2_1");

        // Spherical: h1_0(x) = -i exp(ix)/x exactly, and h2 = conj(h1).
        let x = 2.3;
        let got = get(call_ok("sph_hankel_h1", &[n(0.0), n(x)]));
        let want = Cx::I * -1.0 * Cx::from_polar(1.0 / x, x);
        assert!((got - want).abs() < 1e-14, "h1_0: {got:?} vs {want:?}");
        let h2v = get(call_ok("sph_hankel_h2", &[n(0.0), n(x)]));
        assert!((h2v - got.conj()).abs() < 1e-15, "h2_0 != conj(h1_0)");
        // The spherical Wronskian h1 h2' - h1' h2 = -2i/x^2.
        let h1p = get(call_ok("sph_hankel_h1_prime", &[n(2.0), n(x)]));
        let h2p = get(call_ok("sph_hankel_h2_prime", &[n(2.0), n(x)]));
        let h1v = get(call_ok("sph_hankel_h1", &[n(2.0), n(x)]));
        let h2v = get(call_ok("sph_hankel_h2", &[n(2.0), n(x)]));
        let w = h1v * h2p - h1p * h2v;
        assert!((w - Cx::I * (-2.0 / (x * x))).abs() < 1e-12, "spherical Wronskian");

        // The _z forms refuse a fractional order; the _nu forms take one.
        assert!(call_err("hankel_h1_z", &[n(0.5), n(3.0)]).contains("whole number"));
        let _ = call_ok("hankel_h1_nu", &[n(0.5), n(3.0)]);
        // Singular points report errors rather than infinities.
        assert!(!call_err("hankel_h1_z", &[n(0.0), z(0.0, 0.0)]).is_empty());
        assert!(!call_err("sph_hankel_h1", &[n(0.0), n(0.0)]).is_empty());
    }

    /// The scaled entry points, checked by what makes them worth
    /// having: values the plain forms get wrong, and values the plain
    /// forms cannot represent at all.
    #[test]
    fn scaled_entry_points_are_reachable() {
        use sf::complex::Complex64 as Cx;
        let z = |re: f64, im: f64| Value::Complex(Cx::new(re, im));
        let get = |v: Value| match v {
            Value::Complex(c) => c,
            Value::Num(r) => Cx::real(r),
            other => panic!("expected complex, got {other:?}"),
        };

        // On the real axis exp(-|Im z|) is 1, so the scaled Y IS Y.
        let got = get(call_ok("bessel_y_scaled", &[n(0.0), n(40.0)]));
        let want = sf::cephes::cephes64::yv(0.0, 40.0);
        assert!((got.re - want).abs() < 1e-13, "Y_0(40): {} vs {want}", got.re);
        // The plain form used to be wrong in its first digit here,
        // which was the entire reason the scaled ones existed. Stage 19
        // fixed it at the source, so the two now agree — asserted, so a
        // regression is caught from both sides.
        let plain = get(call_ok("bessel_y_z", &[n(0.0), n(40.0)]));
        assert!(
            (plain.re - want).abs() <= 1e-12 * want.abs(),
            "bessel_y_z(0, 40) should now be right too: {}",
            plain.re
        );

        // exp(x) K_{1/2}(x) = sqrt(pi/2x) exactly, at a magnitude where
        // the unscaled K_{1/2} is far below the smallest f64.
        let got = get(call_ok("bessel_k_scaled", &[n(0.5), n(1.0e6)]));
        let want = (std::f64::consts::PI / 2.0e6).sqrt();
        assert!((got.re - want).abs() < 1e-15, "exp(x)K_1/2(1e6): {} vs {want}", got.re);
        assert_eq!(sf::cephes::cephes64::k0(2000.0), 0.0, "K_0(2000) must underflow");
        assert!(get(call_ok("bessel_k_scaled", &[n(0.0), n(2000.0)])).re > 0.0);

        // exp(-iz) H1_{1/2}(z) = -i sqrt(2/(pi z)), 700 nepers above the
        // real axis where J and Y are each about e^700.
        let zz = Cx::new(3.0, 700.0);
        let got = get(call_ok("hankel_h1_scaled", &[n(0.5), z(zz.re, zz.im)]));
        let want = Cx::I * -1.0
            * (Cx::real(2.0 / std::f64::consts::PI) * zz.inv()).powf(0.5);
        assert!((got - want).abs() / want.abs() < 1e-13, "H1s: {got:?} vs {want:?}");
        // The plain form cannot even be evaluated there — its
        // ingredients leave f64 range — so it errors out.
        assert!(
            !call_err("hankel_h1_z", &[n(0.0), z(zz.re, zz.im)]).is_empty(),
            "the plain H1 is expected to be unusable at Im z = 700"
        );

        // exp(-x) I_0(x) where I_0 itself overflows.
        assert!(sf::cephes::cephes64::i0(1000.0).is_infinite());
        let got = get(call_ok("bessel_i_scaled", &[n(0.0), n(1000.0)]));
        let lead = 1.0 / (2.0 * std::f64::consts::PI * 1000.0).sqrt();
        assert!((got.re / lead - 1.0 - 1.0 / 8000.0).abs() < 1e-6, "I_0 scaled = {}", got.re);

        // The order recurrence, past where Cephes kn overflows.
        assert!(sf::cephes::cephes64::kn(40, 25.0).is_infinite());
        assert!(get(call_ok("bessel_k_scaled", &[n(40.0), n(25.0)])).re > 0.0);

        // J scaled agrees with Cephes on the real axis.
        let got = get(call_ok("bessel_j_scaled", &[n(0.0), n(1000.0)]));
        assert!((got.re - sf::cephes::cephes64::j0(1000.0)).abs() < 1e-13);
        // H2 is reachable and is the conjugate of H1 on the real axis.
        let a = get(call_ok("hankel_h1_scaled", &[n(1.0), n(30.0)]));
        let b = get(call_ok("hankel_h2_scaled", &[n(1.0), n(30.0)]));
        assert!((b - a.conj()).abs() < 1e-13, "H2s != conj(H1s) on the real axis");

        // Singular and unreachable points report errors.
        assert!(!call_err("bessel_k_scaled", &[n(0.0), z(0.0, 0.0)]).is_empty());
        // Since the large-order expansions arrived, this point no
        // longer fails for want of a method — the expansion determines
        // it, and it is the f64 that cannot carry it. The message says
        // which, and quotes the logarithm.
        let e = call_err("bessel_j_scaled", &[n(400.5), n(25.0)]);
        assert!(e.contains("outside f64 range"), "unhelpful message: {e}");
        assert!(e.contains("-992"), "should quote the logarithm: {e}");
        // The genuine no-method refusal is still reachable.
        let e = call_err("bessel_i_scaled", &[n(4000.0), z(1e-6, 300.0)]);
        assert!(e.contains("neither method"), "unhelpful message: {e}");
    }

    /// Complex order, reachable from the language, and the complex
    /// gamma that made it possible.
    #[test]
    fn complex_order_and_gamma_are_reachable() {
        use sf::complex::Complex64 as Cx;
        let z = |re: f64, im: f64| Value::Complex(Cx::new(re, im));
        let get = |v: Value| match v {
            Value::Complex(c) => c,
            Value::Num(r) => Cx::real(r),
            other => panic!("expected complex, got {other:?}"),
        };

        // |Gamma(1+iy)|^2 = pi y / sinh(pi y), a closed form on the
        // imaginary axis with nothing but elementary functions on the
        // right.
        for y in [0.5_f64, 2.0, 7.0] {
            let g = get(call_ok("gamma_z", &[z(1.0, y)]));
            let want = std::f64::consts::PI * y / (std::f64::consts::PI * y).sinh();
            assert!(
                (g.norm_sqr() - want).abs() <= 1e-11 * want,
                "|Gamma(1+{y}i)|^2 = {} vs {want}",
                g.norm_sqr()
            );
        }
        // 1/Gamma is entire: exactly zero at the poles.
        assert_eq!(get(call_ok("rgamma_z", &[n(-3.0)])), Cx::ZERO);
        // ln Gamma is defined where Gamma has left f64 range.
        assert!(call_err("gamma_z", &[n(200.0)]).contains("overflow"));
        let l = get(call_ok("ln_gamma_z", &[n(200.0)]));
        assert!((l.re - sf::cephes::cephes64::lgam(200.0)).abs() < 1e-10);

        // A real order still reaches exactly what it always did.
        let a = get(call_ok("bessel_j_nu", &[n(0.5), n(2.0)]));
        let want = (2.0 / (std::f64::consts::PI * 2.0)).sqrt() * 2.0_f64.sin();
        assert!((a.re - want).abs() < 1e-13, "J_1/2(2) = {a:?}");

        // A complex order, judged by the J-Y Wronskian, whose right-hand
        // side does not involve the order at all.
        let (nu, zz) = (Cx::new(1.0, 2.0), Cx::new(3.0, 0.5));
        let j0 = get(call_ok("bessel_j_nu", &[z(nu.re, nu.im), z(zz.re, zz.im)]));
        let j1 = get(call_ok("bessel_j_nu", &[z(nu.re + 1.0, nu.im), z(zz.re, zz.im)]));
        let y0 = get(call_ok("bessel_y_nu", &[z(nu.re, nu.im), z(zz.re, zz.im)]));
        let y1 = get(call_ok("bessel_y_nu", &[z(nu.re + 1.0, nu.im), z(zz.re, zz.im)]));
        let w = j1 * y0 - j0 * y1;
        let want = zz.inv() * (2.0 / std::f64::consts::PI);
        assert!((w - want).abs() < 1e-11, "Wronskian at complex order: {w:?}");

        // K of imaginary order is real, and even in the order.
        let a = get(call_ok("bessel_k_nu", &[z(0.0, 1.0), n(2.0)]));
        let b = get(call_ok("bessel_k_nu", &[z(0.0, -1.0), n(2.0)]));
        assert!(a.im.abs() < 1e-13, "K_i(2) should be real, got {a:?}");
        assert!((a - b).abs() < 1e-13, "K_-i != K_i");

        assert!(!call_err("bessel_y_nu", &[z(1.0, 1.0), z(0.0, 0.0)]).is_empty());
        assert!(!call_err("ln_gamma_z", &[n(0.0)]).is_empty(), "pole at 0");
    }

    /// `airy_z`, checked by the Wronskian `Ai Bi' - Ai' Bi = 1/pi`,
    /// which is exact and elementary — and by the closed forms at the
    /// origin.
    #[test]
    fn airy_at_complex_argument_is_reachable() {
        use sf::complex::Complex64 as Cx;
        let four = |v: Value| match v {
            Value::List(l) if l.len() == 4 => {
                let g = |x: &Value| match x {
                    Value::Complex(c) => *c,
                    Value::Num(r) => Cx::real(*r),
                    other => panic!("expected complex, got {other:?}"),
                };
                [g(&l[0]), g(&l[1]), g(&l[2]), g(&l[3])]
            }
            other => panic!("expected four values, got {other:?}"),
        };
        for (re, im) in [(0.0, 0.0), (2.0, -3.0), (-8.0, 0.0), (-4.0, 5.0), (30.0, 0.0)] {
            let v = four(call_ok("airy_z", &[Value::Complex(Cx::new(re, im))]));
            let w = v[0] * v[3] - v[1] * v[2];
            let want = Cx::real(1.0 / std::f64::consts::PI);
            // Scaled by the largest term, as everywhere else in this
            // crate: the Wronskian is a DIFFERENCE, so dividing by 1/pi
            // alone measures its own cancellation as well as the
            // routine's error. Unscaled, `z = -4 + 5i` reads 1.4e-7
            // while the values there are good to 1e-11.
            let scale = (v[0] * v[3]).abs() + (v[1] * v[2]).abs();
            assert!(
                (w - want).abs() / scale.max(want.abs()) <= 1e-9,
                "Wronskian at {re}+{im}i: {w:?}"
            );
        }
        // Ai(0) = 3^(-2/3)/Gamma(2/3), and Bi(0) = sqrt(3) Ai(0).
        let v = four(call_ok("airy_z", &[n(0.0)]));
        let g23 = get_re(call_ok("gamma_z", &[n(2.0 / 3.0)]));
        assert!((v[0].re - 3.0_f64.powf(-2.0 / 3.0) / g23).abs() < 1e-13);
        assert!((v[2].re - v[0].re * 3.0_f64.sqrt()).abs() < 1e-15);
        // Past |z| ~ 90 off the real axis the dominant solution leaves
        // f64, and that is reported rather than returned as infinity.
        assert!(!call_err("airy_z", &[Value::Complex(Cx::new(-190.0, -60.0))]).is_empty());
    }

    fn get_re(v: Value) -> f64 {
        match v {
            Value::Complex(c) => c.re,
            Value::Num(r) => r,
            other => panic!("expected a number, got {other:?}"),
        }
    }

    #[test]
    fn unknown_names_fall_through() {
        assert!(call("dot", &[]).is_none(), "must not shadow a core builtin");
        assert!(call("nonesuch", &[]).is_none());
    }

    #[test]
    fn values_match_the_library() {
        // P_2(x) = (3x^2-1)/2
        assert!((as_f(call_ok("legendre_p", &[n(2.0), n(0.5)])) - (-0.125)).abs() < 1e-14);
        // H_3(x) = 8x^3 - 12x  ->  H_3(1) = -4
        assert!((as_f(call_ok("hermite_h", &[n(3.0), n(1.0)])) + 4.0).abs() < 1e-13);
        // T_n(cos t) = cos(n t)
        let t = 0.7_f64;
        let got = as_f(call_ok("chebyshev_t", &[n(5.0), n(t.cos())]));
        assert!((got - (5.0 * t).cos()).abs() < 1e-13);
        // j_0(x) = sin(x)/x
        let got = as_f(call_ok("sph_j", &[n(0.0), n(1.3)]));
        assert!((got - (1.3_f64).sin() / 1.3).abs() < 1e-14);
        // J_0 at its first zero
        assert!(as_f(call_ok("bessel_j", &[n(0.0), n(2.404_825_557_695_773)])).abs() < 1e-12);
    }

    /// A non-integral order is a mistake, and must be reported rather
    /// than truncated into a confident wrong answer.
    #[test]
    fn fractional_orders_are_rejected_not_truncated() {
        let e = call_err("hermite_h", &[n(2.5), n(1.0)]);
        assert!(e.contains("whole number"), "got: {e}");
        // and the truncated call would have succeeded, which is the point
        assert!(call("hermite_h", &[n(2.0), n(1.0)]).unwrap().is_ok());
    }

    #[test]
    fn library_errors_reach_the_user() {
        // negative order
        assert!(call_err("bessel_j", &[n(-1.0), n(1.0)]).contains("order"));
        assert!(!call_err("legendre_p", &[n(-1.0), n(0.5)]).is_empty());
        // |m| > l violates the associated Legendre selection rule
        assert!(!call_err("assoc_legendre_p", &[n(1.0), n(3.0), n(0.5)]).is_empty());
        // NOT an error: P_n is a polynomial, defined for ALL real x —
        // only the orthogonality interval is [-1, 1]. P_2(5) = 37.
        assert!((as_f(call_ok("legendre_p", &[n(2.0), n(5.0)])) - 37.0).abs() < 1e-12);
    }

    #[test]
    fn list_and_matrix_shapes() {
        // bessel_j_array returns n_max+1 entries
        match call_ok("bessel_j_array", &[n(4.0), n(2.0)]) {
            Value::List(v) => assert_eq!(v.len(), 5),
            other => panic!("expected a list, got {other:?}"),
        }
        // eigenvalues of diag(1,2,3), ascending
        let m = Value::List(vec![
            Value::List(vec![n(1.0), n(0.0), n(0.0)]),
            Value::List(vec![n(0.0), n(2.0), n(0.0)]),
            Value::List(vec![n(0.0), n(0.0), n(3.0)]),
        ]);
        match call_ok("eigenvalues", &[m]) {
            Value::List(v) => {
                assert_eq!(v.len(), 3);
                assert!((as_f(v[0].clone()) - 1.0).abs() < 1e-12);
                assert!((as_f(v[2].clone()) - 3.0).abs() < 1e-12);
            }
            other => panic!("expected a list, got {other:?}"),
        }
        // a ragged matrix is refused by shape, not by the eigensolver
        let ragged = Value::List(vec![
            Value::List(vec![n(1.0), n(0.0)]),
            Value::List(vec![n(0.0)]),
        ]);
        assert!(call_err("eigenvalues", &[ragged]).contains("square"));
    }

    #[test]
    fn solve_tridiag_round_trips() {
        // [[2,1,0],[1,2,1],[0,1,2]] x = [1,2,3]  ->  [0.5, 0, 1.5]
        let z = |v: Vec<f64>| Value::List(v.into_iter().map(Value::Num).collect());
        let x = call_ok(
            "solve_tridiag",
            &[
                z(vec![0.0, 1.0, 1.0]),
                z(vec![2.0, 2.0, 2.0]),
                z(vec![1.0, 1.0, 0.0]),
                z(vec![1.0, 2.0, 3.0]),
            ],
        );
        match x {
            Value::List(v) => {
                assert!((as_f(v[0].clone()) - 0.5).abs() < 1e-13);
                assert!(as_f(v[1].clone()).abs() < 1e-13);
                assert!((as_f(v[2].clone()) - 1.5).abs() < 1e-13);
            }
            other => panic!("expected a list, got {other:?}"),
        }
    }

    /// The bracket literal is overloaded (3 -> vector, 4 -> quaternion,
    /// else list). Every shape must work as a numeric list, or a user
    /// typing a perfectly ordinary 4x4 matrix gets told about
    /// quaternions. Both real bugs, found by driving the actual binary.
    #[test]
    fn overloaded_bracket_shapes_all_work_as_lists() {
        use physical_object::linalg::{Quat, Vec3};
        // 3 entries -> Vec3
        let v3 = Value::Vec3(Vec3 { x: 1.0, y: 2.0, z: 3.0 });
        assert_eq!(as_num_list("t", 0, &v3).unwrap(), vec![1.0, 2.0, 3.0]);
        // 4 entries -> Quat, unpacked w-first = the order typed
        let q = Value::Quat(Quat { w: 1.0, x: 2.0, y: 3.0, z: 4.0 });
        assert_eq!(as_num_list("t", 0, &q).unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
        // anything else -> List
        let l = Value::List(vec![n(1.0), n(2.0)]);
        assert_eq!(as_num_list("t", 0, &l).unwrap(), vec![1.0, 2.0]);
        // and a 4x4 matrix built from 4-entry rows must be accepted
        let row = |a: f64, b: f64, c: f64, d: f64| Value::Quat(Quat { w: a, x: b, y: c, z: d });
        let m = Value::List(vec![
            row(2.0, -1.0, 0.0, 0.0),
            row(-1.0, 2.0, -1.0, 0.0),
            row(0.0, -1.0, 2.0, -1.0),
            row(0.0, 0.0, -1.0, 2.0),
        ]);
        let ev = as_matrix("t", 0, &m).unwrap();
        assert_eq!(ev.len(), 4);
        assert_eq!(ev[0].len(), 4);
    }

    /// Complex-argument Bessel, reachable only because the language has
    /// a complex value type. Checked against identities rather than a
    /// table: real argument must reproduce the real routine, and
    /// `J_n(iy) = i^n I_n(y)`.
    #[test]
    fn complex_argument_bessel_is_reachable() {
        use sf::complex::Complex64 as Cx;
        let z = |re: f64, im: f64| Value::Complex(Cx::new(re, im));
        let get = |v: Value| match v {
            Value::Complex(c) => c,
            Value::Num(r) => Cx::real(r),
            other => panic!("expected complex, got {other:?}"),
        };
        // real argument reproduces the real routine: J_0 at its first zero
        let v = get(call_ok("bessel_j_z", &[n(0.0), z(2.404_825_557_695_773, 0.0)]));
        assert!(v.abs() < 1e-12, "J_0 at its first zero = {v:?}");
        // a real number promotes to complex
        let v = get(call_ok("bessel_j_z", &[n(0.0), n(2.404_825_557_695_773)]));
        assert!(v.abs() < 1e-12, "real argument should promote");
        // J_n(i y) = i^n I_n(y): for n = 0 that is real and equals I_0
        let y = 1.5_f64;
        let j0 = get(call_ok("bessel_j_z", &[n(0.0), z(0.0, y)]));
        let i0v = get(call_ok("bessel_i_z", &[n(0.0), z(y, 0.0)]));
        assert!(
            (j0 - i0v).abs() < 1e-10,
            "J_0(i*{y}) = {j0:?} should equal I_0({y}) = {i0v:?}"
        );
        // Y and K: pinned by the Wronskian J_{n+1} Y_n - J_n Y_{n+1}
        // = 2/(pi z), which is elementary on the right-hand side.
        let zz = Cx::new(1.6, 0.9);
        let j0 = get(call_ok("bessel_j_z", &[n(0.0), z(zz.re, zz.im)]));
        let j1 = get(call_ok("bessel_j_z", &[n(1.0), z(zz.re, zz.im)]));
        let y0 = get(call_ok("bessel_y_z", &[n(0.0), z(zz.re, zz.im)]));
        let y1 = get(call_ok("bessel_y_z", &[n(1.0), z(zz.re, zz.im)]));
        let w = j1 * y0 - j0 * y1;
        let want = zz.inv() * (2.0 / std::f64::consts::PI);
        assert!((w - want).abs() < 1e-10, "Wronskian {w:?} vs {want:?}");
        // K via the I-K Wronskian I_0 K_1 + I_1 K_0 = 1/z
        let i0v = get(call_ok("bessel_i_z", &[n(0.0), z(zz.re, zz.im)]));
        let i1v = get(call_ok("bessel_i_z", &[n(1.0), z(zz.re, zz.im)]));
        let k0v = get(call_ok("bessel_k_z", &[n(0.0), z(zz.re, zz.im)]));
        let k1v = get(call_ok("bessel_k_z", &[n(1.0), z(zz.re, zz.im)]));
        let w2 = i0v * k1v + i1v * k0v;
        assert!((w2 - zz.inv()).abs() < 1e-9, "I-K Wronskian {w2:?} vs 1/z");
        // Y and K are singular at the origin
        assert!(!call_err("bessel_y_z", &[n(0.0), z(0.0, 0.0)]).is_empty());
        assert!(!call_err("bessel_k_z", &[n(0.0), z(0.0, 0.0)]).is_empty());
        // the order must still be a whole number
        assert!(call_err("bessel_j_z", &[n(1.5), z(1.0, 1.0)]).contains("whole number"));
    }

    /// Complex values: literal, arithmetic, and the solvers they were
    /// added for.
    #[test]
    fn complex_values_work_end_to_end() {
        use sf::complex::Complex64 as C;
        let c = |re: f64, im: f64| Value::Complex(C::new(re, im));
        // a real band with a complex diagonal, real rhs
        let x = call_ok(
            "solve_tridiag_c",
            &[
                Value::List(vec![n(0.0), n(1.0), n(1.0)]),
                Value::List(vec![c(0.0, 1.0), c(0.0, 1.0), c(0.0, 1.0)]),
                Value::List(vec![n(1.0), n(1.0), n(0.0)]),
                Value::List(vec![n(1.0), n(0.0), n(0.0)]),
            ],
        );
        // verify by substitution: A x must reproduce the rhs
        let xs: Vec<C> = match x {
            Value::List(v) => v
                .into_iter()
                .map(|e| match e {
                    Value::Complex(z) => z,
                    Value::Num(r) => C::real(r),
                    other => panic!("expected complex, got {other:?}"),
                })
                .collect(),
            other => panic!("expected a list, got {other:?}"),
        };
        let d = C::new(0.0, 1.0);
        let r0 = d * xs[0] + C::real(1.0) * xs[1];
        let r1 = C::real(1.0) * xs[0] + d * xs[1] + C::real(1.0) * xs[2];
        let r2 = C::real(1.0) * xs[1] + d * xs[2];
        assert!((r0 - C::ONE).abs() < 1e-12, "row 0 residual");
        assert!(r1.abs() < 1e-12, "row 1 residual");
        assert!(r2.abs() < 1e-12, "row 2 residual");
    }

    /// Angular-momentum coupling reaches the language, including
    /// half-integer arguments, which is why these take plain numbers
    /// rather than going through `as_int`.
    #[test]
    fn wigner_symbols_are_reachable() {
        // (1 1 0; 0 0 0) = -1/sqrt(3)
        let v = as_f(call_ok("wigner_3j", &[n(1.0), n(1.0), n(0.0), n(0.0), n(0.0), n(0.0)]));
        assert!((v + 1.0 / 3.0_f64.sqrt()).abs() < 1e-13);
        // half-integer arguments must be ACCEPTED, not rejected as
        // "not whole numbers" -- spins are half-integral
        let v = as_f(call_ok(
            "clebsch_gordan",
            &[n(0.5), n(0.5), n(0.5), n(-0.5), n(1.0), n(0.0)],
        ));
        assert!((v - 1.0 / 2.0_f64.sqrt()).abs() < 1e-13);
        // {1 1 1; 1 1 1} = 1/6
        let v = as_f(call_ok("wigner_6j", &[n(1.0), n(1.0), n(1.0), n(1.0), n(1.0), n(1.0)]));
        assert!((v - 1.0 / 6.0).abs() < 1e-13);
        // 9-j: {1 1 1; 1 1 1; 1 1 0} reduces to (1/3) * {1 1 1; 1 1 1}
        // = (1/3)(1/6) = 1/18 by the zero-argument closed form
        let v = as_f(call_ok(
            "wigner_9j",
            &[n(1.0), n(1.0), n(1.0), n(1.0), n(1.0), n(1.0), n(1.0), n(1.0), n(0.0)],
        ));
        assert!((v - 1.0 / 18.0).abs() < 1e-12, "9j = {v}, want 1/18");
        // a broken triangle is 0, not an error
        let v = as_f(call_ok(
            "wigner_9j",
            &[n(1.0), n(1.0), n(9.0), n(1.0), n(1.0), n(1.0), n(1.0), n(1.0), n(1.0)],
        ));
        assert_eq!(v, 0.0);
        // a genuinely invalid spin is still an error
        assert!(!call_err("wigner_3j", &[n(0.3), n(1.0), n(1.0), n(0.0), n(0.0), n(0.0)]).is_empty());
    }

    #[test]
    fn sph_harm_returns_re_and_im() {
        match call_ok("sph_harm", &[n(1.0), n(0.0), n(0.6), n(0.0)]) {
            Value::List(v) => {
                assert_eq!(v.len(), 2, "expected [re, im]");
                // Y_1^0 is real, so the imaginary part must vanish
                assert!(as_f(v[1].clone()).abs() < 1e-14);
            }
            other => panic!("expected a list, got {other:?}"),
        }
    }
}
