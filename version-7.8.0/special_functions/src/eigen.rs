//! Dense real-symmetric eigenvalue problems by the cyclic Jacobi method.
//!
//! Matrix mechanics *is* diagonalisation: put a Hamiltonian in a basis,
//! diagonalise, read off the spectrum and the eigenstates. Without this
//! a simulator is confined to the handful of analytically soluble
//! systems, which is why this module exists.
//!
//! # Why Jacobi rather than QR
//!
//! The Householder-tridiagonalisation + QL route is asymptotically
//! faster (O(n³) with a small constant versus Jacobi's several sweeps
//! of O(n³)). Jacobi is chosen anyway because:
//!
//! * it is **unconditionally convergent** — every rotation strictly
//!   decreases the off-diagonal norm, so there is no failure mode to
//!   handle beyond an iteration cap;
//! * it delivers **eigenvectors orthogonal to machine precision** by
//!   construction, since they are accumulated as a product of exact
//!   plane rotations;
//! * it is **more accurate for small eigenvalues**, which is exactly
//!   the regime of a ground-state energy;
//! * it is ~200 lines with no pivoting or deflation logic to get wrong.
//!
//! For the matrix sizes a notebook-driven simulator will realistically
//! diagonalise (n of order 10²–10³), the constant factor is irrelevant
//! next to being obviously correct.

// A Jacobi sweep walks a matrix by index, mutating rows and columns
// through the SAME index in the same iteration. clippy's
// `needless_range_loop` wants iterator forms that cannot express that
// without splitting borrows, and the result would be markedly harder to
// check against the published algorithm. The index loops stay.
#![allow(clippy::needless_range_loop)]

/// Diagonalise a real symmetric matrix.
///
/// `a` is an `n x n` symmetric matrix given row-wise. Returns
/// `(eigenvalues, eigenvectors)` with eigenvalues in **ascending**
/// order; `eigenvectors[k]` is the unit eigenvector belonging to
/// `eigenvalues[k]`.
///
/// The input is not modified.
///
/// # Errors
/// Empty or ragged input, non-square shape, any non-finite entry,
/// asymmetry beyond a small tolerance, or failure to converge within
/// the sweep cap.
///
/// # Examples
/// ```
/// use special_functions::eigen::jacobi_eigen;
/// // [[2,1],[1,2]] has eigenvalues 1 and 3
/// let a = vec![vec![2.0, 1.0], vec![1.0, 2.0]];
/// let (vals, vecs) = jacobi_eigen(&a).unwrap();
/// assert!((vals[0] - 1.0).abs() < 1e-12);
/// assert!((vals[1] - 3.0).abs() < 1e-12);
/// // eigenvectors are unit length
/// let n0: f64 = vecs[0].iter().map(|c| c * c).sum();
/// assert!((n0 - 1.0).abs() < 1e-12);
/// ```
pub fn jacobi_eigen(a: &[Vec<f64>]) -> Result<(Vec<f64>, Vec<Vec<f64>>), String> {
    let n = a.len();
    if n == 0 {
        return Err("jacobi_eigen: matrix is empty".to_string());
    }
    for (i, row) in a.iter().enumerate() {
        if row.len() != n {
            return Err(format!(
                "jacobi_eigen: matrix must be square; row {i} has {} entries, expected {n}",
                row.len()
            ));
        }
        for (j, &v) in row.iter().enumerate() {
            if !v.is_finite() {
                return Err(format!("jacobi_eigen: entry ({i},{j}) is not finite: {v}"));
            }
        }
    }
    // Symmetry check, scaled to the size of the entries.
    let scale = a
        .iter()
        .flat_map(|r| r.iter())
        .fold(0.0_f64, |m, v| m.max(v.abs()))
        .max(1.0);
    for i in 0..n {
        for j in (i + 1)..n {
            if (a[i][j] - a[j][i]).abs() > 1.0e-12 * scale {
                return Err(format!(
                    "jacobi_eigen: matrix is not symmetric at ({i},{j}): {} vs {}",
                    a[i][j], a[j][i]
                ));
            }
        }
    }

    let mut m: Vec<Vec<f64>> = a.to_vec();
    // Eigenvector accumulator, starts as the identity.
    let mut v = vec![vec![0.0_f64; n]; n];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.0;
    }

    let off = |m: &Vec<Vec<f64>>| -> f64 {
        let mut s = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                s += m[i][j] * m[i][j];
            }
        }
        s
    };

    let eps = 1.0e-30_f64 * scale * scale;
    const MAX_SWEEPS: usize = 100;
    let mut converged = n == 1;

    for _sweep in 0..MAX_SWEEPS {
        if off(&m) <= eps {
            converged = true;
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                if m[p][q].abs() <= 1.0e-300 {
                    continue;
                }
                // Rotation that annihilates m[p][q]: the standard stable
                // form, taking the smaller root of t^2 + 2*theta*t - 1.
                let theta = (m[q][q] - m[p][p]) / (2.0 * m[p][q]);
                let t = if theta >= 0.0 {
                    1.0 / (theta + (1.0 + theta * theta).sqrt())
                } else {
                    -1.0 / (-theta + (1.0 + theta * theta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                // Apply J^T M J.
                for k in 0..n {
                    let mkp = m[k][p];
                    let mkq = m[k][q];
                    m[k][p] = c * mkp - s * mkq;
                    m[k][q] = s * mkp + c * mkq;
                }
                for k in 0..n {
                    let mpk = m[p][k];
                    let mqk = m[q][k];
                    m[p][k] = c * mpk - s * mqk;
                    m[q][k] = s * mpk + c * mqk;
                }
                // Accumulate the rotation into the eigenvector matrix.
                for row in v.iter_mut() {
                    let vp = row[p];
                    let vq = row[q];
                    row[p] = c * vp - s * vq;
                    row[q] = s * vp + c * vq;
                }
            }
        }
    }
    if !converged && off(&m) > eps.max(1.0e-20 * scale * scale) {
        return Err(format!(
            "jacobi_eigen: failed to converge in {MAX_SWEEPS} sweeps (off-diagonal norm^2 = {})",
            off(&m)
        ));
    }

    // Columns of `v` are the eigenvectors; transpose into rows and sort
    // by ascending eigenvalue.
    let mut pairs: Vec<(f64, Vec<f64>)> = (0..n)
        .map(|k| (m[k][k], (0..n).map(|i| v[i][k]).collect::<Vec<f64>>()))
        .collect();
    pairs.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));

    let vals: Vec<f64> = pairs.iter().map(|p| p.0).collect();
    let vecs: Vec<Vec<f64>> = pairs
        .into_iter()
        .map(|(_, mut e)| {
            // Fix the sign convention: make the largest-magnitude
            // component positive, so results are reproducible.
            let mut idx = 0usize;
            let mut best = 0.0_f64;
            for (i, &c) in e.iter().enumerate() {
                if c.abs() > best {
                    best = c.abs();
                    idx = i;
                }
            }
            if e[idx] < 0.0 {
                for c in e.iter_mut() {
                    *c = -*c;
                }
            }
            e
        })
        .collect();
    Ok((vals, vecs))
}

/// Eigenvalues only, ascending — a convenience wrapper when the
/// eigenvectors are not needed (e.g. reading off a spectrum).
///
/// # Examples
/// ```
/// use special_functions::eigen::eigenvalues;
/// let a = vec![vec![3.0, 0.0], vec![0.0, -1.0]];
/// let e = eigenvalues(&a).unwrap();
/// assert!((e[0] + 1.0).abs() < 1e-14 && (e[1] - 3.0).abs() < 1e-14);
/// ```
pub fn eigenvalues(a: &[Vec<f64>]) -> Result<Vec<f64>, String> {
    Ok(jacobi_eigen(a)?.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rel_err;
    use std::f64::consts::PI;

    fn matvec(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
        a.iter()
            .map(|row| row.iter().zip(x).map(|(r, v)| r * v).sum())
            .collect()
    }

    #[test]
    fn diagonal_matrix_returns_its_diagonal_sorted() {
        let a = vec![
            vec![5.0, 0.0, 0.0],
            vec![0.0, -2.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let (vals, _) = jacobi_eigen(&a).unwrap();
        assert!(rel_err(vals[0], -2.0) < 1e-14);
        assert!(rel_err(vals[1], 1.0) < 1e-14);
        assert!(rel_err(vals[2], 5.0) < 1e-14);
    }

    #[test]
    fn two_by_two_known_spectrum() {
        let a = vec![vec![2.0, 1.0], vec![1.0, 2.0]];
        let (vals, vecs) = jacobi_eigen(&a).unwrap();
        assert!(rel_err(vals[0], 1.0) < 1e-13);
        assert!(rel_err(vals[1], 3.0) < 1e-13);
        // eigenvectors are (1,-1)/sqrt2 and (1,1)/sqrt2
        let s = 1.0 / 2.0_f64.sqrt();
        assert!((vecs[1][0].abs() - s).abs() < 1e-13);
        assert!((vecs[1][1].abs() - s).abs() < 1e-13);
    }

    /// The defining property: A v = lambda v for every pair.
    #[test]
    fn eigenpairs_satisfy_the_eigenvalue_equation() {
        let a = vec![
            vec![4.0, 1.0, -2.0, 2.0],
            vec![1.0, 2.0, 0.0, 1.0],
            vec![-2.0, 0.0, 3.0, -2.0],
            vec![2.0, 1.0, -2.0, -1.0],
        ];
        let (vals, vecs) = jacobi_eigen(&a).unwrap();
        for (lam, v) in vals.iter().zip(&vecs) {
            let av = matvec(&a, v);
            for i in 0..v.len() {
                assert!(
                    (av[i] - lam * v[i]).abs() < 1e-10,
                    "A v != lambda v at {i}: {} vs {}",
                    av[i],
                    lam * v[i]
                );
            }
        }
    }

    #[test]
    fn eigenvectors_are_orthonormal() {
        let a = vec![
            vec![4.0, 1.0, -2.0, 2.0],
            vec![1.0, 2.0, 0.0, 1.0],
            vec![-2.0, 0.0, 3.0, -2.0],
            vec![2.0, 1.0, -2.0, -1.0],
        ];
        let (_, vecs) = jacobi_eigen(&a).unwrap();
        for i in 0..vecs.len() {
            for j in 0..vecs.len() {
                let dot: f64 = vecs[i].iter().zip(&vecs[j]).map(|(x, y)| x * y).sum();
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((dot - want).abs() < 1e-12, "<{i}|{j}> = {dot}");
            }
        }
    }

    /// Trace and determinant are invariants: sum and product of the
    /// eigenvalues must reproduce them.
    #[test]
    fn trace_and_determinant_invariants() {
        let a = vec![
            vec![4.0, 1.0, -2.0],
            vec![1.0, 2.0, 0.0],
            vec![-2.0, 0.0, 3.0],
        ];
        let (vals, _) = jacobi_eigen(&a).unwrap();
        let trace: f64 = (0..3).map(|i| a[i][i]).sum();
        assert!(rel_err(vals.iter().sum::<f64>(), trace) < 1e-12);
        // det by cofactor expansion
        let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
        assert!(rel_err(vals.iter().product::<f64>(), det) < 1e-11);
    }

    /// The 1-D Laplacian with Dirichlet ends has the exact spectrum
    /// `4 sin^2(k pi / 2(n+1))`. This is the particle-in-a-box
    /// Hamiltonian, so it checks the solver on the very problem the
    /// module exists to serve.
    #[test]
    fn tridiagonal_laplacian_matches_its_analytic_spectrum() {
        let n = 40usize;
        let mut a = vec![vec![0.0; n]; n];
        for i in 0..n {
            a[i][i] = 2.0;
            if i + 1 < n {
                a[i][i + 1] = -1.0;
                a[i + 1][i] = -1.0;
            }
        }
        let vals = eigenvalues(&a).unwrap();
        for (k, &lam) in vals.iter().enumerate() {
            let exact = 4.0 * ((k + 1) as f64 * PI / (2.0 * (n + 1) as f64)).sin().powi(2);
            assert!(rel_err(lam, exact) < 1e-10, "mode {k}: {lam} vs {exact}");
        }
    }

    /// Particle in a box, end to end: discretise -d^2/dx^2 on [0, L]
    /// and confirm the lowest levels approach the analytic
    /// `E_n = n^2 pi^2 / L^2` as the grid refines.
    #[test]
    fn particle_in_a_box_energies_converge_to_the_analytic_result() {
        let l = 1.0_f64;
        let n = 200usize;
        let h = l / (n + 1) as f64;
        let mut a = vec![vec![0.0; n]; n];
        for i in 0..n {
            a[i][i] = 2.0 / (h * h);
            if i + 1 < n {
                a[i][i + 1] = -1.0 / (h * h);
                a[i + 1][i] = -1.0 / (h * h);
            }
        }
        let vals = eigenvalues(&a).unwrap();
        for k in 0..4 {
            let kk = (k + 1) as f64;
            let exact = (kk * PI / l).powi(2);
            // A second-order stencil has a KNOWN leading error:
            //     E_k^FD / E_k^exact = [sin(x)/x]^2 with x = k pi h / 2,
            // i.e. a relative deficit of (k pi h)^2 / 12. Asserting that
            // law (10% margin) is far stronger than a fixed tolerance —
            // it pins the convergence ORDER, not just closeness.
            let predicted = (kk * PI * h).powi(2) / 12.0;
            let observed = (exact - vals[k]) / exact;
            assert!(
                observed > 0.0,
                "level {}: the discrete eigenvalue must lie BELOW the continuum value",
                k + 1
            );
            assert!(
                (observed - predicted).abs() < 0.1 * predicted,
                "level {}: discretisation error {observed:.3e} does not follow the \
                 predicted (k pi h)^2/12 = {predicted:.3e}",
                k + 1
            );
        }
    }

    #[test]
    fn invalid_input_is_reported() {
        assert!(jacobi_eigen(&[]).is_err(), "empty");
        assert!(
            jacobi_eigen(&[vec![1.0, 2.0]]).is_err(),
            "non-square (1x2)"
        );
        assert!(
            jacobi_eigen(&[vec![1.0, 2.0], vec![3.0, 4.0]]).is_err(),
            "asymmetric"
        );
        assert!(
            jacobi_eigen(&[vec![f64::NAN, 0.0], vec![0.0, 1.0]]).is_err(),
            "non-finite entry"
        );
        // 1x1 is legal
        let (v, _) = jacobi_eigen(&[vec![7.0]]).unwrap();
        assert!(rel_err(v[0], 7.0) < 1e-15);
    }
}
