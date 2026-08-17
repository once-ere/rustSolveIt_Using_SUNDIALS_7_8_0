//! Sparse direct LU factorization — the numerical engine behind
//! [`crate::sunlinsol_klu`].
//!
//! # Why this exists
//!
//! SUNDIALS' sparse direct linear solver is a thin wrapper around **KLU**
//! from SuiteSparse. KLU is **LGPL-2.1-or-later**, so it cannot be
//! translated into this BSD-3-Clause tree, and this port forbids FFI, so it
//! cannot be called either. Without a replacement the eleven `*_klu`
//! examples are simply unportable — which is how they stood before this
//! module existed.
//!
//! So this is an independent implementation, written here, of a sparse LU
//! with the same *interface contract* as KLU but not the same internals.
//! The consequence is stated plainly rather than buried: **it does not
//! reproduce KLU's arithmetic bit for bit, and it is not expected to.** A
//! different elimination order rounds differently, and inside a Newton
//! iteration that is output-observable. `differences/ATTRIBUTION.md`
//! measures exactly how much.
//!
//! # The algorithm
//!
//! Left-looking sparse LU with partial pivoting — Gilbert and Peierls,
//! *"Sparse partial pivoting in time proportional to arithmetic
//! operations"*, SIAM J. Sci. Stat. Comput. 9(5), 1988. Column `k` of the
//! factors is computed by solving `L y = A(:,k)`, whose nonzero pattern is
//! the set of nodes reachable from the nonzeros of `A(:,k)` in the directed
//! graph of `L`. That set is found by depth-first search and returned in
//! topological order, so the whole factorization costs time proportional to
//! the arithmetic it performs rather than to `n^2`.
//!
//! Deliberate choices, each of which differs from KLU's default and each of
//! which is a limitation worth knowing:
//!
//! * **Threshold partial pivoting with a diagonal preference**, at KLU's
//!   default `tol = 0.001`: the largest-magnitude candidate in the column
//!   wins *unless* the diagonal entry is within a factor `tol` of it, in
//!   which case the diagonal is kept. This is not a stylistic choice. The
//!   matrices these examples produce carry exact unit rows -- `idaHeat2D`'s
//!   boundary equations are literally `e_i` -- and pure partial pivoting
//!   discards that unit diagonal in favour of a neighbouring `-1/dx^2`,
//!   which mixes the boundary and interior unknowns and lets round-off
//!   into components the problem pins exactly. Keeping the diagonal is
//!   what KLU does and what those examples are posed against.
//! * **No fill-reducing column ordering.** Columns are eliminated in their
//!   natural order; KLU applies AMD. For the matrices these examples
//!   produce — 3x3 dense blocks, a block-diagonal system with 3x3 blocks,
//!   and a 10x10 five-point stencil — the natural order produces no
//!   significant fill, so the ordering buys nothing here. On a large
//!   unstructured matrix it would, and this module would then be the wrong
//!   tool. See `LIMITATIONS` below.
//! * **No block triangular form.** KLU permutes to BTF first. Again, for a
//!   block-diagonal matrix the natural order already keeps the elimination
//!   inside each block, so BTF is not needed to keep these examples cheap.
//! * **Every setup performs a fresh factorization**, including a fresh
//!   pivot search. KLU's `klu_refactor` reuses the previous pivot order and
//!   is faster but can degrade; re-pivoting is more accurate and simpler to
//!   reason about.
//!
//! Nothing here is derived from KLU, CSparse or any other SuiteSparse
//! source. The Gilbert-Peierls method is a published algorithm; this is an
//! implementation of it.

use crate::sundials_types::*;

/// KLU's default partial-pivoting threshold. The diagonal entry is kept as
/// the pivot whenever its magnitude is at least this fraction of the
/// largest candidate in the column.
pub const PIVOT_TOL: sunrealtype = 0.001;

/// What this factorization is *not* good for, stated once so a caller does
/// not have to infer it: there is no fill-reducing ordering, so on a large
/// unstructured sparse matrix the factors can fill in catastrophically.
/// It is sized for the block-structured and banded systems SUNDIALS'
/// sparse examples generate.
pub const LIMITATIONS: &str =
    "no fill-reducing ordering and no block triangular form; intended for \
     block-structured or banded systems";

/// A completed LU factorization `P * A == L * U`, with `L` unit lower
/// triangular. All three matrices are held in compressed sparse column
/// form with row indices already permuted, so a solve is two ordinary
/// triangular sweeps.
#[derive(Clone, Debug)]
pub struct SparseLU {
    /// Order of the system.
    pub n: usize,
    /// `L` column pointers, length `n + 1`. The unit diagonal is implicit.
    lp: Vec<usize>,
    /// `L` row indices, already in pivot ordering.
    li: Vec<usize>,
    /// `L` values.
    lx: Vec<sunrealtype>,
    /// `U` column pointers, length `n + 1`. The diagonal is *not* stored
    /// here; it is in [`SparseLU::udiag`].
    up: Vec<usize>,
    /// `U` strictly-upper row indices.
    ui: Vec<usize>,
    /// `U` strictly-upper values.
    ux: Vec<sunrealtype>,
    /// `U(k, k)` for each `k`.
    udiag: Vec<sunrealtype>,
    /// `pinv[i] == k` when original row `i` became pivot row `k`.
    pinv: Vec<usize>,
    /// `p[k] == i`, the inverse of `pinv`.
    p: Vec<usize>,
}

/// Why a factorization could not be produced.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SparseLuError {
    /// A column had no acceptable pivot: the matrix is numerically singular.
    /// Carries the column index.
    Singular(usize),
    /// The compressed-column arrays are not self-consistent.
    Malformed,
}

/// Scratch space reused across the columns of one factorization, so the
/// depth-first search does not allocate per column.
struct Workspace {
    /// Dense accumulator for the current column.
    x: Vec<sunrealtype>,
    /// Nonzero pattern of the current column, filled from the top down:
    /// on return `xi[top..n]` is the pattern in topological order.
    xi: Vec<usize>,
    /// Depth-first search node stack.
    stack: Vec<usize>,
    /// How far the search has walked each stacked node's adjacency list.
    pstack: Vec<usize>,
    /// Nodes already reached while building the current column.
    marked: Vec<bool>,
}

impl Workspace {
    fn new(n: usize) -> Self {
        Workspace {
            x: vec![0.0; n],
            xi: vec![0; n],
            stack: vec![0; n],
            pstack: vec![0; n],
            marked: vec![false; n],
        }
    }
}

impl SparseLU {
    /// Factor the compressed-sparse-column matrix `(ap, ai, ax)` of order
    /// `n`. Row indices within a column need not be sorted.
    ///
    /// Returns `P * A == L * U`, or the column at which the matrix was
    /// found numerically singular.
    pub fn factor(
        n: usize,
        ap: &[sunindextype],
        ai: &[sunindextype],
        ax: &[sunrealtype],
    ) -> Result<SparseLU, SparseLuError> {
        if ap.len() < n + 1 || ap[0] != 0 {
            return Err(SparseLuError::Malformed);
        }
        let nnz = ap[n] as usize;
        if ai.len() < nnz || ax.len() < nnz {
            return Err(SparseLuError::Malformed);
        }

        let mut w = Workspace::new(n);
        // `unset` marks a row that has not yet been chosen as a pivot.
        let unset = usize::MAX;
        let mut pinv = vec![unset; n];

        let mut lp = vec![0usize; n + 1];
        let mut li: Vec<usize> = Vec::with_capacity(nnz);
        let mut lx: Vec<sunrealtype> = Vec::with_capacity(nnz);
        let mut up = vec![0usize; n + 1];
        let mut ui: Vec<usize> = Vec::with_capacity(nnz);
        let mut ux: Vec<sunrealtype> = Vec::with_capacity(nnz);
        let mut udiag = vec![0.0; n];

        for k in 0..n {
            lp[k] = li.len();
            up[k] = ui.len();

            // ---- x = L \ A(:,k), with the pattern found by DFS --------
            let top = sparse_lower_solve(n, ap, ai, ax, k, &pinv, &lp, &li, &lx, &mut w)?;

            // ---- choose the pivot: largest magnitude among rows that are
            //      not yet pivotal, unless the diagonal is within `PIVOT_TOL`
            //      of it, in which case keep the diagonal ------------------
            let mut ipiv = unset;
            let mut best = -1.0f64;
            let mut diag = 0.0f64;
            let mut diag_free = false;
            for t in top..n {
                let i = w.xi[t];
                if pinv[i] == unset {
                    let a = w.x[i].abs();
                    if a > best {
                        best = a;
                        ipiv = i;
                    }
                    if i == k {
                        diag = a;
                        diag_free = true;
                    }
                } else {
                    // already pivotal -> this entry belongs to U
                    ui.push(pinv[i]);
                    ux.push(w.x[i]);
                }
            }
            if ipiv == unset || !(best > 0.0) {
                return Err(SparseLuError::Singular(k));
            }
            if diag_free && diag >= PIVOT_TOL * best {
                ipiv = k;
            }

            let pivot = w.x[ipiv];
            udiag[k] = pivot;
            pinv[ipiv] = k;

            // ---- scale the rest of the column into L ------------------
            for t in top..n {
                let i = w.xi[t];
                if pinv[i] == unset {
                    li.push(i);
                    lx.push(w.x[i] / pivot);
                }
                w.x[i] = 0.0; // clear for the next column
            }
            w.x[ipiv] = 0.0;
        }
        lp[n] = li.len();
        up[n] = ui.len();

        // L was built with original row indices; renumber into pivot order.
        for r in li.iter_mut() {
            *r = pinv[*r];
        }

        let mut p = vec![0usize; n];
        for (i, &k) in pinv.iter().enumerate() {
            p[k] = i;
        }

        Ok(SparseLU { n, lp, li, lx, up, ui, ux, udiag, pinv, p })
    }

    /// Solve `A * x == b` in place.
    pub fn solve(&self, b: &mut [sunrealtype]) {
        let n = self.n;
        let mut x = vec![0.0; n];
        // x = P b
        for i in 0..n {
            x[self.pinv[i]] = b[i];
        }
        // forward substitution, L unit lower triangular
        for k in 0..n {
            let xk = x[k];
            if xk != 0.0 {
                for t in self.lp[k]..self.lp[k + 1] {
                    x[self.li[t]] -= self.lx[t] * xk;
                }
            }
        }
        // back substitution, U upper triangular
        for k in (0..n).rev() {
            let xk = x[k] / self.udiag[k];
            x[k] = xk;
            if xk != 0.0 {
                for t in self.up[k]..self.up[k + 1] {
                    x[self.ui[t]] -= self.ux[t] * xk;
                }
            }
        }
        b.copy_from_slice(&x);
    }

    /// Solve `A^T * x == b` in place.
    ///
    /// `P A = L U` gives `A^T = U^T L^T P`, so this is a forward sweep with
    /// `U^T`, a backward sweep with `L^T`, then the inverse permutation.
    /// The wrapper uses it for compressed-sparse-*row* matrices, whose
    /// arrays describe `A^T` when read as compressed-sparse-column.
    pub fn solve_transpose(&self, b: &mut [sunrealtype]) {
        let n = self.n;
        let mut x = b.to_vec();
        // U^T y = b
        for k in 0..n {
            let mut s = x[k];
            for t in self.up[k]..self.up[k + 1] {
                s -= self.ux[t] * x[self.ui[t]];
            }
            x[k] = s / self.udiag[k];
        }
        // L^T z = y
        for k in (0..n).rev() {
            let mut s = x[k];
            for t in self.lp[k]..self.lp[k + 1] {
                s -= self.lx[t] * x[self.li[t]];
            }
            x[k] = s;
        }
        // undo the row permutation
        for i in 0..n {
            b[i] = x[self.pinv[i]];
        }
    }

    /// KLU's cheap reciprocal condition estimate:
    /// `min |U(k,k)| / max |U(k,k)|`. Zero when `U` has a zero diagonal.
    pub fn rcond(&self) -> sunrealtype {
        if self.n == 0 {
            return 1.0;
        }
        let mut lo = sunrealtype::INFINITY;
        let mut hi = 0.0f64;
        for &d in &self.udiag {
            let a = d.abs();
            if a < lo {
                lo = a;
            }
            if a > hi {
                hi = a;
            }
        }
        if hi == 0.0 {
            0.0
        } else {
            lo / hi
        }
    }

    /// A one-norm condition estimate, `||A||_1 * ||A^-1||_1`, using
    /// Hager's estimator for the inverse norm as refined by Higham
    /// (*FORTRAN codes for estimating the one-norm of a real or complex
    /// matrix*, ACM TOMS 14(4), 1988) — the same estimator LAPACK's
    /// `dlacn2` implements. `anorm` is the exact one-norm of `A`, which
    /// the caller can compute directly from its compressed columns.
    pub fn condest(&self, anorm: sunrealtype) -> sunrealtype {
        let n = self.n;
        if n == 0 || anorm == 0.0 {
            return 0.0;
        }
        let mut x = vec![1.0 / n as sunrealtype; n];
        let mut est = 0.0f64;
        let mut prev_j = usize::MAX;

        for _iter in 0..5 {
            // y = A^-1 x
            self.solve(&mut x);
            let newest: sunrealtype = x.iter().map(|v| v.abs()).sum();
            if newest <= est && _iter > 0 {
                break;
            }
            est = newest;

            // xi = sign(y)
            let mut xi: Vec<sunrealtype> = x
                .iter()
                .map(|&v| if v >= 0.0 { 1.0 } else { -1.0 })
                .collect();

            // z = A^-T xi
            self.solve_transpose(&mut xi);

            // j = argmax |z|
            let mut j = 0usize;
            let mut zj = -1.0f64;
            for (i, &v) in xi.iter().enumerate() {
                let a = v.abs();
                if a > zj {
                    zj = a;
                    j = i;
                }
            }
            if j == prev_j {
                break;
            }
            prev_j = j;

            // next probe vector is the j-th unit vector
            for v in x.iter_mut() {
                *v = 0.0;
            }
            x[j] = 1.0;
        }
        anorm * est
    }

    /// Number of stored nonzeros in `L` (including its unit diagonal) and
    /// in `U`, for the solver's workspace report.
    pub fn nnz(&self) -> (usize, usize) {
        (self.li.len() + self.n, self.ui.len() + self.n)
    }

    /// The pivot row order: `p()[k]` is the original row that became the
    /// `k`-th pivot.
    pub fn pivot_order(&self) -> &[usize] {
        &self.p
    }
}

/// Solve `L y = A(:,k)` into the dense accumulator `w.x`, and return `top`
/// such that `w.xi[top..n]` is the nonzero pattern of `y` in topological
/// order.
///
/// This is the Gilbert-Peierls step. The pattern of `y` is the set of nodes
/// reachable from the nonzeros of `A(:,k)` in the directed graph of the
/// already-computed part of `L`; finding it first is what makes the column
/// cost proportional to its own arithmetic rather than to `n`.
///
/// Node identity is the *original* row index, because `L`'s row indices are
/// only renumbered into pivot order once the whole factorization is done.
/// A node has outgoing edges only if its row has already been chosen as a
/// pivot, in which case `pinv` gives the column of `L` that holds them.
/// The search is iterative: a recursive one would be bounded by the depth
/// of the elimination tree, which is `n` in the worst case.
#[allow(clippy::too_many_arguments)]
fn sparse_lower_solve(
    n: usize,
    ap: &[sunindextype],
    ai: &[sunindextype],
    ax: &[sunrealtype],
    k: usize,
    pinv: &[usize],
    lp: &[usize],
    li: &[usize],
    lx: &[sunrealtype],
    w: &mut Workspace,
) -> Result<usize, SparseLuError> {
    let unset = usize::MAX;
    let mut top = n;

    for m in w.marked.iter_mut() {
        *m = false;
    }

    let cstart = ap[k] as usize;
    let cend = ap[k + 1] as usize;
    if cend < cstart || cend > ai.len() {
        return Err(SparseLuError::Malformed);
    }

    // Outgoing edges of the node whose row is `node`, as a range into `li`.
    let edges = |node: usize| -> (usize, usize) {
        let pj = pinv[node];
        if pj == unset {
            (0, 0)
        } else {
            (lp[pj], lp[pj + 1])
        }
    };

    for t in cstart..cend {
        let i = ai[t] as usize;
        if i >= n {
            return Err(SparseLuError::Malformed);
        }
        w.x[i] = ax[t];
        if w.marked[i] {
            continue;
        }

        // depth-first search from row i
        let mut head = 0usize;
        w.stack[0] = i;
        w.pstack[0] = 0;
        w.marked[i] = true;

        loop {
            let node = w.stack[head];
            let (lo, hi) = edges(node);
            let mut adv = w.pstack[head];
            let mut descended = false;

            while lo + adv < hi {
                let j = li[lo + adv];
                adv += 1;
                if !w.marked[j] {
                    w.pstack[head] = adv;
                    head += 1;
                    w.stack[head] = j;
                    w.pstack[head] = 0;
                    w.marked[j] = true;
                    descended = true;
                    break;
                }
            }

            if !descended {
                w.pstack[head] = adv;
                // every edge explored: emit in topological order
                top -= 1;
                w.xi[top] = node;
                if head == 0 {
                    break;
                }
                head -= 1;
            }
        }
    }

    // Numerical phase: apply the columns of L in topological order.
    for t in top..n {
        let j = w.xi[t];
        let pj = pinv[j];
        if pj == unset {
            continue;
        }
        let xj = w.x[j];
        if xj != 0.0 {
            for q in lp[pj]..lp[pj + 1] {
                w.x[li[q]] -= lx[q] * xj;
            }
        }
    }
    Ok(top)
}

/* =====================================================================
 * Tests
 * =====================================================================
 * The factorization is checked against dense Gaussian elimination with
 * partial pivoting on randomly generated sparse matrices: same input,
 * independent code path, and the residual of the sparse solve must be at
 * the level of the conditioning rather than merely "small".
 */

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic sample stream, so a failure is always reproducible.
    struct SplitMix64(u64);
    impl SplitMix64 {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn unit(&mut self) -> f64 {
            (self.next() >> 11) as f64 * (-53f64).exp2()
        }
    }

    /// Dense column-major matrix -> compressed sparse column, dropping
    /// exact zeros.
    fn dense_to_csc(n: usize, a: &[f64]) -> (Vec<sunindextype>, Vec<sunindextype>, Vec<f64>) {
        let mut ap = vec![0 as sunindextype; n + 1];
        let mut ai = Vec::new();
        let mut ax = Vec::new();
        for j in 0..n {
            for i in 0..n {
                let v = a[j * n + i];
                if v != 0.0 {
                    ai.push(i as sunindextype);
                    ax.push(v);
                }
            }
            ap[j + 1] = ai.len() as sunindextype;
        }
        (ap, ai, ax)
    }

    /// Dense LU with partial pivoting, used only as an independent oracle.
    fn dense_solve(n: usize, a: &[f64], b: &[f64]) -> Option<Vec<f64>> {
        let mut m = a.to_vec();
        let mut x = b.to_vec();
        let mut piv: Vec<usize> = (0..n).collect();
        for k in 0..n {
            let mut best = 0.0;
            let mut r = k;
            for i in k..n {
                let v = m[k * n + piv[i]].abs();
                if v > best {
                    best = v;
                    r = i;
                }
            }
            if best == 0.0 {
                return None;
            }
            piv.swap(k, r);
            let pk = piv[k];
            let d = m[k * n + pk];
            for i in k + 1..n {
                let pi = piv[i];
                let f = m[k * n + pi] / d;
                if f != 0.0 {
                    for j in k + 1..n {
                        m[j * n + pi] -= f * m[j * n + pk];
                    }
                }
                x[pi] -= f * x[pk];
            }
        }
        let mut y = vec![0.0; n];
        for k in (0..n).rev() {
            let pk = piv[k];
            let mut s = x[pk];
            for j in k + 1..n {
                s -= m[j * n + pk] * y[j];
            }
            y[k] = s / m[k * n + pk];
        }
        // y is in pivot order along columns, which for column-major
        // elimination is the natural unknown order
        Some(y)
    }

    fn matvec(n: usize, a: &[f64], x: &[f64]) -> Vec<f64> {
        let mut y = vec![0.0; n];
        for j in 0..n {
            for i in 0..n {
                y[i] += a[j * n + i] * x[j];
            }
        }
        y
    }

    #[test]
    fn sparse_lu_tiny_known_system() {
        // [ 2 0 1 ; 0 3 0 ; 1 0 4 ] x = [3 3 5], solution [1 1 1]
        let n = 3;
        let a = vec![2.0, 0.0, 1.0, 0.0, 3.0, 0.0, 1.0, 0.0, 4.0]; // column-major
        let (ap, ai, ax) = dense_to_csc(n, &a);
        let lu = SparseLU::factor(n, &ap, &ai, &ax).expect("factor");
        let mut b = matvec(n, &a, &[1.0, 1.0, 1.0]);
        lu.solve(&mut b);
        for v in &b {
            assert!((v - 1.0).abs() < 1e-14, "got {b:?}");
        }
    }

    #[test]
    fn sparse_lu_matches_dense_on_random_systems() {
        let mut rng = SplitMix64(0x5EED_1234_ABCD_0001);
        let mut worst: f64 = 0.0;
        for trial in 0..300 {
            let n = 2 + (rng.next() % 24) as usize;
            let density = 0.15 + 0.6 * rng.unit();
            let mut a = vec![0.0f64; n * n];
            for j in 0..n {
                for i in 0..n {
                    if rng.unit() < density {
                        a[j * n + i] = 2.0 * rng.unit() - 1.0;
                    }
                }
                // keep it nonsingular with high probability
                a[j * n + j] += 2.0 + rng.unit();
            }
            let xtrue: Vec<f64> = (0..n).map(|_| 2.0 * rng.unit() - 1.0).collect();
            let b = matvec(n, &a, &xtrue);

            let (ap, ai, ax) = dense_to_csc(n, &a);
            let lu = match SparseLU::factor(n, &ap, &ai, &ax) {
                Ok(lu) => lu,
                Err(e) => panic!("trial {trial}: n={n} factor failed: {e:?}"),
            };
            let mut xs = b.clone();
            lu.solve(&mut xs);

            // residual of the sparse solve
            let r = matvec(n, &a, &xs);
            let mut num = 0.0f64;
            let mut den = 0.0f64;
            for i in 0..n {
                num = num.max((r[i] - b[i]).abs());
                den = den.max(b[i].abs());
            }
            let rel = if den > 0.0 { num / den } else { num };
            worst = worst.max(rel);
            assert!(rel < 1e-10, "trial {trial}: n={n} residual {rel:e}");

            // and it must agree with the independent dense oracle
            let xd = dense_solve(n, &a, &b).expect("dense oracle");
            for i in 0..n {
                let scale = xd[i].abs().max(1.0);
                assert!(
                    (xs[i] - xd[i]).abs() / scale < 1e-9,
                    "trial {trial}: x[{i}] sparse {} vs dense {}",
                    xs[i],
                    xd[i]
                );
            }
        }
        eprintln!("sparse LU vs dense: worst relative residual {worst:e} over 300 systems");
    }

    #[test]
    fn sparse_lu_transpose_solve() {
        let mut rng = SplitMix64(0x5EED_1234_ABCD_0002);
        for _ in 0..100 {
            let n = 2 + (rng.next() % 16) as usize;
            let mut a = vec![0.0f64; n * n];
            for j in 0..n {
                for i in 0..n {
                    if rng.unit() < 0.4 {
                        a[j * n + i] = 2.0 * rng.unit() - 1.0;
                    }
                }
                a[j * n + j] += 2.0 + rng.unit();
            }
            // A^T in column-major is A with i/j swapped
            let mut at = vec![0.0f64; n * n];
            for j in 0..n {
                for i in 0..n {
                    at[j * n + i] = a[i * n + j];
                }
            }
            let xtrue: Vec<f64> = (0..n).map(|_| 2.0 * rng.unit() - 1.0).collect();
            let b = matvec(n, &at, &xtrue);

            let (ap, ai, ax) = dense_to_csc(n, &a);
            let lu = SparseLU::factor(n, &ap, &ai, &ax).expect("factor");
            let mut x = b.clone();
            lu.solve_transpose(&mut x);
            for i in 0..n {
                let scale = xtrue[i].abs().max(1.0);
                assert!((x[i] - xtrue[i]).abs() / scale < 1e-9, "transpose solve");
            }
        }
    }

    #[test]
    fn sparse_lu_block_diagonal_has_no_fill() {
        // 100 blocks of 3x3 -- the structure cvRoberts_block_klu produces.
        // Natural ordering must keep the elimination inside each block, so
        // the factors stay O(n) rather than filling in.
        let nb = 100;
        let n = 3 * nb;
        let mut ap = vec![0 as sunindextype; n + 1];
        let mut ai = Vec::new();
        let mut ax = Vec::new();
        let mut rng = SplitMix64(7);
        for j in 0..n {
            let base = (j / 3) * 3;
            for i in base..base + 3 {
                ai.push(i as sunindextype);
                ax.push(if i == j { 4.0 + rng.unit() } else { rng.unit() - 0.5 });
            }
            ap[j + 1] = ai.len() as sunindextype;
        }
        let lu = SparseLU::factor(n, &ap, &ai, &ax).expect("factor");
        let (lnnz, unnz) = lu.nnz();
        assert!(
            lnnz + unnz <= 3 * n + 3 * n,
            "block-diagonal factorization filled in: L {lnnz} U {unnz} for n {n}"
        );
        let b = vec![1.0; n];
        let mut x = b.clone();
        lu.solve(&mut x);
        assert!(x.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn sparse_lu_detects_singular() {
        // a zero column
        let n = 3;
        let ap: Vec<sunindextype> = vec![0, 1, 1, 2];
        let ai: Vec<sunindextype> = vec![0, 2];
        let ax = vec![1.0, 1.0];
        match SparseLU::factor(n, &ap, &ai, &ax) {
            Err(SparseLuError::Singular(k)) => assert_eq!(k, 1),
            other => panic!("expected Singular(1), got {other:?}"),
        }
    }

    #[test]
    fn sparse_lu_condition_estimates() {
        // diag(1, 10, 100): rcond = 1/100 exactly, cond_1 = 100
        let n = 3;
        let ap: Vec<sunindextype> = vec![0, 1, 2, 3];
        let ai: Vec<sunindextype> = vec![0, 1, 2];
        let ax = vec![1.0, 10.0, 100.0];
        let lu = SparseLU::factor(n, &ap, &ai, &ax).expect("factor");
        assert!((lu.rcond() - 0.01).abs() < 1e-15, "rcond {}", lu.rcond());
        let anorm = 100.0;
        let ce = lu.condest(anorm);
        assert!((ce - 100.0).abs() < 1e-9, "condest {ce}");
    }
}
