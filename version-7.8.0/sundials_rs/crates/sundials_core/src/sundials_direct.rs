//! Port of `src/sundials/sundials_direct.c` +
//! `include/sundials/sundials_direct.h` (legacy SUNDlsMat type).
//!
//! `sunrealtype** a` (array of column pointers) maps to
//! `&mut [&mut [sunrealtype]]`; `dls_cols` builds the column views from
//! the flat column-major storage.

use std::cell::RefCell;
use std::rc::Rc;

use crate::sundials_math::{SUNMAX, SUNMIN};
use crate::sundials_types::*;
use crate::sundials_utils::{sun_format_e, SUNFile};

pub const SUNDIALS_DENSE: i32 = 1;
pub const SUNDIALS_BAND: i32 = 2;

pub struct SUNDlsMat_ {
    pub type_: i32,
    pub M: sunindextype,
    pub N: sunindextype,
    pub ldim: sunindextype,
    pub mu: sunindextype,
    pub ml: sunindextype,
    pub s_mu: sunindextype,
    pub data: Vec<sunrealtype>,
    pub ldata: sunindextype,
}

pub type SUNDlsMat = Rc<RefCell<SUNDlsMat_>>;

/// Build the C `cols` column-pointer view over flat column-major storage.
pub fn dls_cols(data: &mut [sunrealtype], ldim: sunindextype) -> Vec<&mut [sunrealtype]> {
    data.chunks_mut(ldim as usize).collect()
}

pub fn SUNDlsMat_NewDenseMat(M: sunindextype, N: sunindextype) -> Option<SUNDlsMat> {
    if M <= 0 || N <= 0 {
        return None;
    }
    Some(Rc::new(RefCell::new(SUNDlsMat_ {
        type_: SUNDIALS_DENSE,
        M,
        N,
        ldim: M,
        mu: 0,
        ml: 0,
        s_mu: 0,
        data: vec![0.0; (M * N) as usize],
        ldata: M * N,
    })))
}

pub fn SUNDlsMat_NewBandMat(
    N: sunindextype,
    mu: sunindextype,
    ml: sunindextype,
    smu: sunindextype,
) -> Option<SUNDlsMat> {
    if N <= 0 {
        return None;
    }
    let colSize = smu + ml + 1;
    Some(Rc::new(RefCell::new(SUNDlsMat_ {
        type_: SUNDIALS_BAND,
        M: N,
        N,
        mu,
        ml,
        s_mu: smu,
        ldim: colSize,
        data: vec![0.0; (N * colSize) as usize],
        ldata: N * colSize,
    })))
}

pub fn SUNDlsMat_DestroyMat(A: SUNDlsMat) {
    drop(A);
}

pub fn SUNDlsMat_AddIdentity(A: &SUNDlsMat) {
    let mut a = A.borrow_mut();
    match a.type_ {
        SUNDIALS_DENSE => {
            let (n, ldim) = (a.N, a.ldim);
            for i in 0..n {
                a.data[(i * ldim + i) as usize] += 1.0;
            }
        }
        SUNDIALS_BAND => {
            let (m, ldim, s_mu) = (a.M, a.ldim, a.s_mu);
            for i in 0..m {
                a.data[(i * ldim + s_mu) as usize] += 1.0;
            }
        }
        _ => {}
    }
}

pub fn SUNDlsMat_SetToZero(A: &SUNDlsMat) {
    let mut a = A.borrow_mut();
    match a.type_ {
        SUNDIALS_DENSE => {
            let (m, n, ldim) = (a.M, a.N, a.ldim);
            for j in 0..n {
                for i in 0..m {
                    a.data[(j * ldim + i) as usize] = 0.0;
                }
            }
        }
        SUNDIALS_BAND => {
            let (m, mu, ml, s_mu, ldim) = (a.M, a.mu, a.ml, a.s_mu, a.ldim);
            let colSize = mu + ml + 1;
            for j in 0..m {
                let base = j * ldim + s_mu - mu;
                for i in 0..colSize {
                    a.data[(base + i) as usize] = 0.0;
                }
            }
        }
        _ => {}
    }
}

pub fn SUNDlsMat_PrintMat(A: &SUNDlsMat, outfile: &SUNFile) {
    let a = A.borrow();
    match a.type_ {
        SUNDIALS_DENSE => {
            outfile.write_str("\n");
            for i in 0..a.M {
                for j in 0..a.N {
                    outfile.write_str(&format!(
                        "{}  ",
                        sun_format_e(a.data[(j * a.ldim + i) as usize])
                    ));
                }
                outfile.write_str("\n");
            }
        }
        SUNDIALS_BAND => {
            outfile.write_str("\n");
            for i in 0..a.N {
                let start = SUNMAX(0, i - a.ml);
                let finish = SUNMIN(a.N - 1, i + a.mu);
                for _ in 0..start {
                    outfile.write_str(&format!("{:12}  ", ""));
                }
                for j in start..=finish {
                    outfile.write_str(&format!(
                        "{}  ",
                        sun_format_e(a.data[(j * a.ldim + i - j + a.s_mu) as usize])
                    ));
                }
                outfile.write_str("\n");
            }
        }
        _ => {}
    }
}
