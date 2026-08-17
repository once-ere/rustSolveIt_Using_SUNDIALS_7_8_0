//! Port of `src/sundials/sundials_matrix.c` +
//! `include/sundials/sundials_matrix.h` (generic SUNMATRIX layer).
//!
//! Same handle model as N_Vector: `SUNMatrix = Rc<_generic_SUNMatrix>`.

use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

use crate::sundials_context::SUNContext;
use crate::sundials_errors::{SUN_ERR_NOT_IMPLEMENTED, SUN_SUCCESS};
use crate::sundials_nvector::N_Vector;
use crate::sundials_types::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SUNMatrix_ID {
    SUNMATRIX_DENSE,
    SUNMATRIX_MAGMADENSE,
    SUNMATRIX_ONEMKLDENSE,
    SUNMATRIX_BAND,
    SUNMATRIX_SPARSE,
    SUNMATRIX_SLUNRLOC,
    SUNMATRIX_CUSPARSE,
    SUNMATRIX_GINKGO,
    SUNMATRIX_GINKGOBATCH,
    SUNMATRIX_KOKKOSDENSE,
    SUNMATRIX_CUSTOM,
}
pub use SUNMatrix_ID::*;

#[derive(Default, Clone)]
pub struct _generic_SUNMatrix_Ops {
    pub getid: Option<fn(&SUNMatrix) -> SUNMatrix_ID>,
    pub clone: Option<fn(&SUNMatrix) -> Option<SUNMatrix>>,
    pub destroy: Option<fn(SUNMatrix)>,
    pub zero: Option<fn(&SUNMatrix) -> SUNErrCode>,
    pub copy: Option<fn(&SUNMatrix, &SUNMatrix) -> SUNErrCode>,
    pub scaleadd: Option<fn(sunrealtype, &SUNMatrix, &SUNMatrix) -> SUNErrCode>,
    pub scaleaddi: Option<fn(sunrealtype, &SUNMatrix) -> SUNErrCode>,
    pub matvecsetup: Option<fn(&SUNMatrix) -> SUNErrCode>,
    pub matvec: Option<fn(&SUNMatrix, &N_Vector, &N_Vector) -> SUNErrCode>,
    pub mathermitiantransposevec: Option<fn(&SUNMatrix, &N_Vector, &N_Vector) -> SUNErrCode>,
    pub space: Option<fn(&SUNMatrix, &mut i64, &mut i64) -> SUNErrCode>,
}

pub type SUNMatrix_Ops = _generic_SUNMatrix_Ops;

pub struct _generic_SUNMatrix {
    pub content: RefCell<Box<dyn Any>>,
    pub ops: RefCell<_generic_SUNMatrix_Ops>,
    pub sunctx: RefCell<SUNContext>,
}

pub type SUNMatrix = Rc<_generic_SUNMatrix>;

pub fn SUNMatNewEmpty(sunctx: &SUNContext) -> Option<SUNMatrix> {
    Some(Rc::new(_generic_SUNMatrix {
        content: RefCell::new(Box::new(())),
        ops: RefCell::new(_generic_SUNMatrix_Ops::default()),
        sunctx: RefCell::new(sunctx.clone()),
    }))
}

pub fn SUNMatFreeEmpty(A: SUNMatrix) {
    drop(A);
}

/// C `SUNMatCopyOps` — note upstream does **not** copy
/// `mathermitiantransposevec`; preserved verbatim.
pub fn SUNMatCopyOps(A: &SUNMatrix, B: &SUNMatrix) -> SUNErrCode {
    let a = A.ops.borrow();
    let mut b = B.ops.borrow_mut();
    b.getid = a.getid;
    b.clone = a.clone;
    b.destroy = a.destroy;
    b.zero = a.zero;
    b.copy = a.copy;
    b.scaleadd = a.scaleadd;
    b.scaleaddi = a.scaleaddi;
    b.matvecsetup = a.matvecsetup;
    b.matvec = a.matvec;
    b.space = a.space;
    0
}

pub fn SUNMatGetID(A: &SUNMatrix) -> SUNMatrix_ID {
    let f = A.ops.borrow().getid.expect("getid");
    f(A)
}

pub fn SUNMatClone(A: &SUNMatrix) -> Option<SUNMatrix> {
    let f = A.ops.borrow().clone.expect("clone");
    let B = f(A);
    if let Some(B) = &B {
        *B.sunctx.borrow_mut() = A.sunctx.borrow().clone();
    }
    B
}

pub fn SUNMatDestroy(A: SUNMatrix) {
    let f = A.ops.borrow().destroy;
    if let Some(f) = f {
        f(A);
        return;
    }
    drop(A);
}

pub fn SUNMatZero(A: &SUNMatrix) -> SUNErrCode {
    let f = A.ops.borrow().zero.expect("zero");
    f(A)
}

pub fn SUNMatCopy(A: &SUNMatrix, B: &SUNMatrix) -> SUNErrCode {
    let f = A.ops.borrow().copy.expect("copy");
    f(A, B)
}

pub fn SUNMatScaleAdd(c: sunrealtype, A: &SUNMatrix, B: &SUNMatrix) -> SUNErrCode {
    let f = A.ops.borrow().scaleadd.expect("scaleadd");
    f(c, A, B)
}

pub fn SUNMatScaleAddI(c: sunrealtype, A: &SUNMatrix) -> SUNErrCode {
    let f = A.ops.borrow().scaleaddi.expect("scaleaddi");
    f(c, A)
}

pub fn SUNMatMatvecSetup(A: &SUNMatrix) -> SUNErrCode {
    let f = A.ops.borrow().matvecsetup;
    match f {
        Some(f) => f(A),
        None => SUN_SUCCESS,
    }
}

pub fn SUNMatMatvec(A: &SUNMatrix, x: &N_Vector, y: &N_Vector) -> SUNErrCode {
    let f = A.ops.borrow().matvec.expect("matvec");
    f(A, x, y)
}

pub fn SUNMatHermitianTransposeVec(A: &SUNMatrix, x: &N_Vector, y: &N_Vector) -> SUNErrCode {
    let f = A.ops.borrow().mathermitiantransposevec;
    match f {
        Some(f) => f(A, x, y),
        None => SUN_ERR_NOT_IMPLEMENTED,
    }
}

pub fn SUNMatSpace(A: &SUNMatrix, lenrw: &mut i64, leniw: &mut i64) -> SUNErrCode {
    let f = A.ops.borrow().space.expect("space");
    f(A, lenrw, leniw)
}
