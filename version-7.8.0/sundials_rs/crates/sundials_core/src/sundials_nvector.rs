//! Port of `src/sundials/sundials_nvector.c` +
//! `include/sundials/sundials_nvector.h` (generic NVECTOR layer).
//!
//! Handle model (ARCHITECTURE.md): `N_Vector = Rc<_generic_N_Vector>`;
//! `content` is the C `void*` (`RefCell<Box<dyn Any>>`), `ops` is the C
//! ops table (`RefCell` because C code overwrites op slots in place).
//! `N_VGetArrayPointer` returns a `RefMut` guard mapped to the data
//! (`sunrealtype*` in C); callers index it like the C pointer and drop it
//! before invoking other ops on the same vector.

use std::any::Any;
use std::cell::{RefCell, RefMut};
use std::rc::Rc;

use crate::sundials_context::SUNContext;
use crate::sundials_errors::SUN_SUCCESS;
use crate::sundials_types::*;
use crate::sundials_utils::SUNFile;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum N_Vector_ID {
    SUNDIALS_NVEC_SERIAL,
    SUNDIALS_NVEC_PARALLEL,
    SUNDIALS_NVEC_OPENMP,
    SUNDIALS_NVEC_PTHREADS,
    SUNDIALS_NVEC_PARHYP,
    SUNDIALS_NVEC_PETSC,
    SUNDIALS_NVEC_CUDA,
    SUNDIALS_NVEC_HIP,
    SUNDIALS_NVEC_SYCL,
    SUNDIALS_NVEC_RAJA,
    SUNDIALS_NVEC_KOKKOS,
    SUNDIALS_NVEC_OPENMPDEV,
    SUNDIALS_NVEC_TRILINOS,
    SUNDIALS_NVEC_MANYVECTOR,
    SUNDIALS_NVEC_MPIMANYVECTOR,
    SUNDIALS_NVEC_MPIPLUSX,
    SUNDIALS_NVEC_CUSTOM,
}
pub use N_Vector_ID::*;

/// Structure containing function pointers to vector operations.
#[derive(Default, Clone)]
pub struct _generic_N_Vector_Ops {
    /* constructors, destructors, and utility operations */
    pub nvgetvectorid: Option<fn(&N_Vector) -> N_Vector_ID>,
    pub nvclone: Option<fn(&N_Vector) -> Option<N_Vector>>,
    pub nvcloneempty: Option<fn(&N_Vector) -> Option<N_Vector>>,
    pub nvdestroy: Option<fn(N_Vector)>,
    pub nvspace: Option<fn(&N_Vector, &mut sunindextype, &mut sunindextype)>,
    pub nvgetarraypointer:
        Option<for<'a> fn(&'a N_Vector) -> Option<RefMut<'a, Vec<sunrealtype>>>>,
    pub nvgetdevicearraypointer:
        Option<for<'a> fn(&'a N_Vector) -> Option<RefMut<'a, Vec<sunrealtype>>>>,
    pub nvsetarraypointer: Option<fn(Vec<sunrealtype>, &N_Vector)>,
    pub nvgetcommunicator: Option<fn(&N_Vector) -> SUNComm>,
    pub nvgetlength: Option<fn(&N_Vector) -> sunindextype>,
    pub nvgetlocallength: Option<fn(&N_Vector) -> sunindextype>,

    /* standard vector operations */
    pub nvlinearsum: Option<fn(sunrealtype, &N_Vector, sunrealtype, &N_Vector, &N_Vector)>,
    pub nvconst: Option<fn(sunrealtype, &N_Vector)>,
    pub nvprod: Option<fn(&N_Vector, &N_Vector, &N_Vector)>,
    pub nvdiv: Option<fn(&N_Vector, &N_Vector, &N_Vector)>,
    pub nvscale: Option<fn(sunrealtype, &N_Vector, &N_Vector)>,
    pub nvabs: Option<fn(&N_Vector, &N_Vector)>,
    pub nvinv: Option<fn(&N_Vector, &N_Vector)>,
    pub nvaddconst: Option<fn(&N_Vector, sunrealtype, &N_Vector)>,
    pub nvdotprod: Option<fn(&N_Vector, &N_Vector) -> sunrealtype>,
    pub nvmaxnorm: Option<fn(&N_Vector) -> sunrealtype>,
    pub nvwrmsnorm: Option<fn(&N_Vector, &N_Vector) -> sunrealtype>,
    pub nvwrmsnormmask: Option<fn(&N_Vector, &N_Vector, &N_Vector) -> sunrealtype>,
    pub nvmin: Option<fn(&N_Vector) -> sunrealtype>,
    pub nvwl2norm: Option<fn(&N_Vector, &N_Vector) -> sunrealtype>,
    pub nvl1norm: Option<fn(&N_Vector) -> sunrealtype>,
    pub nvcompare: Option<fn(sunrealtype, &N_Vector, &N_Vector)>,
    pub nvinvtest: Option<fn(&N_Vector, &N_Vector) -> sunbooleantype>,
    pub nvconstrmask: Option<fn(&N_Vector, &N_Vector, &N_Vector) -> sunbooleantype>,
    pub nvminquotient: Option<fn(&N_Vector, &N_Vector) -> sunrealtype>,

    /* OPTIONAL fused vector operations */
    pub nvlinearcombination:
        Option<fn(i32, &[sunrealtype], &[N_Vector], &N_Vector) -> SUNErrCode>,
    pub nvscaleaddmulti:
        Option<fn(i32, &[sunrealtype], &N_Vector, &[N_Vector], &[N_Vector]) -> SUNErrCode>,
    pub nvdotprodmulti:
        Option<fn(i32, &N_Vector, &[N_Vector], &mut [sunrealtype]) -> SUNErrCode>,

    /* OPTIONAL vector array operations */
    pub nvlinearsumvectorarray: Option<
        fn(i32, sunrealtype, &[N_Vector], sunrealtype, &[N_Vector], &[N_Vector]) -> SUNErrCode,
    >,
    pub nvscalevectorarray:
        Option<fn(i32, &[sunrealtype], &[N_Vector], &[N_Vector]) -> SUNErrCode>,
    pub nvconstvectorarray: Option<fn(i32, sunrealtype, &[N_Vector]) -> SUNErrCode>,
    pub nvwrmsnormvectorarray:
        Option<fn(i32, &[N_Vector], &[N_Vector], &mut [sunrealtype]) -> SUNErrCode>,
    pub nvwrmsnormmaskvectorarray:
        Option<fn(i32, &[N_Vector], &[N_Vector], &N_Vector, &mut [sunrealtype]) -> SUNErrCode>,
    pub nvscaleaddmultivectorarray: Option<
        fn(i32, i32, &[sunrealtype], &[N_Vector], &[Vec<N_Vector>], &[Vec<N_Vector>]) -> SUNErrCode,
    >,
    pub nvlinearcombinationvectorarray:
        Option<fn(i32, i32, &[sunrealtype], &[Vec<N_Vector>], &[N_Vector]) -> SUNErrCode>,

    /* OPTIONAL local reduction kernels (no parallel communication) */
    pub nvdotprodlocal: Option<fn(&N_Vector, &N_Vector) -> sunrealtype>,
    pub nvmaxnormlocal: Option<fn(&N_Vector) -> sunrealtype>,
    pub nvminlocal: Option<fn(&N_Vector) -> sunrealtype>,
    pub nvl1normlocal: Option<fn(&N_Vector) -> sunrealtype>,
    pub nvinvtestlocal: Option<fn(&N_Vector, &N_Vector) -> sunbooleantype>,
    pub nvconstrmasklocal: Option<fn(&N_Vector, &N_Vector, &N_Vector) -> sunbooleantype>,
    pub nvminquotientlocal: Option<fn(&N_Vector, &N_Vector) -> sunrealtype>,
    pub nvwsqrsumlocal: Option<fn(&N_Vector, &N_Vector) -> sunrealtype>,
    pub nvwsqrsummasklocal: Option<fn(&N_Vector, &N_Vector, &N_Vector) -> sunrealtype>,

    /* single buffer reduction operations */
    pub nvdotprodmultilocal:
        Option<fn(i32, &N_Vector, &[N_Vector], &mut [sunrealtype]) -> SUNErrCode>,
    pub nvdotprodmultiallreduce: Option<fn(i32, &N_Vector, &mut [sunrealtype]) -> SUNErrCode>,

    /* XBraid interface operations */
    pub nvbufsize: Option<fn(&N_Vector, &mut sunindextype) -> SUNErrCode>,
    pub nvbufpack: Option<fn(&N_Vector, &mut [sunrealtype]) -> SUNErrCode>,
    pub nvbufunpack: Option<fn(&N_Vector, &[sunrealtype]) -> SUNErrCode>,

    /* Debugging functions */
    pub nvprint: Option<fn(&N_Vector)>,
    pub nvprintfile: Option<fn(&N_Vector, &SUNFile)>,
}

pub type N_Vector_Ops = _generic_N_Vector_Ops;

pub struct _generic_N_Vector {
    pub content: RefCell<Box<dyn Any>>,
    pub ops: RefCell<_generic_N_Vector_Ops>,
    pub sunctx: RefCell<SUNContext>,
}

pub type N_Vector = Rc<_generic_N_Vector>;

/// Create an empty NVector object (all ops NULL, content empty).
pub fn N_VNewEmpty(sunctx: &SUNContext) -> Option<N_Vector> {
    Some(Rc::new(_generic_N_Vector {
        content: RefCell::new(Box::new(())),
        ops: RefCell::new(_generic_N_Vector_Ops::default()),
        sunctx: RefCell::new(sunctx.clone()),
    }))
}

/// Free a generic N_Vector (assumes content is already empty).
pub fn N_VFreeEmpty(v: N_Vector) {
    drop(v);
}

/// Copy a vector 'ops' structure from `w` to `v`.
pub fn N_VCopyOps(w: &N_Vector, v: &N_Vector) -> SUNErrCode {
    *v.ops.borrow_mut() = w.ops.borrow().clone();
    SUN_SUCCESS
}

/* -----------------------------------------------------------------
 * Functions in the 'ops' structure
 * -----------------------------------------------------------------*/

pub fn N_VGetVectorID(w: &N_Vector) -> N_Vector_ID {
    let f = w.ops.borrow().nvgetvectorid.expect("nvgetvectorid");
    f(w)
}

pub fn N_VClone(w: &N_Vector) -> Option<N_Vector> {
    let f = w.ops.borrow().nvclone.expect("nvclone");
    let result = f(w);
    if let Some(result) = &result {
        *result.sunctx.borrow_mut() = w.sunctx.borrow().clone();
    }
    result
}

pub fn N_VCloneEmpty(w: &N_Vector) -> Option<N_Vector> {
    let f = w.ops.borrow().nvcloneempty.expect("nvcloneempty");
    let result = f(w);
    if let Some(result) = &result {
        *result.sunctx.borrow_mut() = w.sunctx.borrow().clone();
    }
    result
}

pub fn N_VDestroy(v: N_Vector) {
    /* if the destroy operation exists use it */
    let f = v.ops.borrow().nvdestroy;
    if let Some(f) = f {
        f(v);
    }
    /* otherwise dropping the handle releases content, ops, and vector */
}

pub fn N_VSpace(v: &N_Vector, lrw: &mut sunindextype, liw: &mut sunindextype) {
    let f = v.ops.borrow().nvspace.expect("nvspace");
    f(v, lrw, liw);
}

pub fn N_VGetArrayPointer(v: &N_Vector) -> Option<RefMut<'_, Vec<sunrealtype>>> {
    let f = v.ops.borrow().nvgetarraypointer;
    match f {
        Some(f) => f(v),
        None => None,
    }
}

pub fn N_VGetDeviceArrayPointer(v: &N_Vector) -> Option<RefMut<'_, Vec<sunrealtype>>> {
    let f = v.ops.borrow().nvgetdevicearraypointer;
    match f {
        Some(f) => f(v),
        None => None,
    }
}

pub fn N_VSetArrayPointer(v_data: Vec<sunrealtype>, v: &N_Vector) {
    let f = v.ops.borrow().nvsetarraypointer;
    if let Some(f) = f {
        f(v_data, v);
    }
}

pub fn N_VGetCommunicator(v: &N_Vector) -> SUNComm {
    let f = v.ops.borrow().nvgetcommunicator;
    match f {
        Some(f) => f(v),
        None => SUN_COMM_NULL,
    }
}

pub fn N_VGetLength(v: &N_Vector) -> sunindextype {
    let f = v.ops.borrow().nvgetlength.expect("nvgetlength");
    f(v)
}

pub fn N_VGetLocalLength(v: &N_Vector) -> sunindextype {
    let f = v.ops.borrow().nvgetlocallength.expect("nvgetlocallength");
    f(v)
}

/* -----------------------------------------------------------------
 * standard vector operations
 * -----------------------------------------------------------------*/

pub fn N_VLinearSum(a: sunrealtype, x: &N_Vector, b: sunrealtype, y: &N_Vector, z: &N_Vector) {
    let f = z.ops.borrow().nvlinearsum.expect("nvlinearsum");
    f(a, x, b, y, z);
}

pub fn N_VConst(c: sunrealtype, z: &N_Vector) {
    let f = z.ops.borrow().nvconst.expect("nvconst");
    f(c, z);
}

pub fn N_VProd(x: &N_Vector, y: &N_Vector, z: &N_Vector) {
    let f = z.ops.borrow().nvprod.expect("nvprod");
    f(x, y, z);
}

pub fn N_VDiv(x: &N_Vector, y: &N_Vector, z: &N_Vector) {
    let f = z.ops.borrow().nvdiv.expect("nvdiv");
    f(x, y, z);
}

pub fn N_VScale(c: sunrealtype, x: &N_Vector, z: &N_Vector) {
    let f = z.ops.borrow().nvscale.expect("nvscale");
    f(c, x, z);
}

pub fn N_VAbs(x: &N_Vector, z: &N_Vector) {
    let f = z.ops.borrow().nvabs.expect("nvabs");
    f(x, z);
}

pub fn N_VInv(x: &N_Vector, z: &N_Vector) {
    let f = z.ops.borrow().nvinv.expect("nvinv");
    f(x, z);
}

pub fn N_VAddConst(x: &N_Vector, b: sunrealtype, z: &N_Vector) {
    let f = z.ops.borrow().nvaddconst.expect("nvaddconst");
    f(x, b, z);
}

pub fn N_VDotProd(x: &N_Vector, y: &N_Vector) -> sunrealtype {
    let f = y.ops.borrow().nvdotprod.expect("nvdotprod");
    f(x, y)
}

pub fn N_VMaxNorm(x: &N_Vector) -> sunrealtype {
    let f = x.ops.borrow().nvmaxnorm.expect("nvmaxnorm");
    f(x)
}

pub fn N_VWrmsNorm(x: &N_Vector, w: &N_Vector) -> sunrealtype {
    let f = x.ops.borrow().nvwrmsnorm.expect("nvwrmsnorm");
    f(x, w)
}

pub fn N_VWrmsNormMask(x: &N_Vector, w: &N_Vector, id: &N_Vector) -> sunrealtype {
    let f = x.ops.borrow().nvwrmsnormmask.expect("nvwrmsnormmask");
    f(x, w, id)
}

pub fn N_VMin(x: &N_Vector) -> sunrealtype {
    let f = x.ops.borrow().nvmin.expect("nvmin");
    f(x)
}

pub fn N_VWL2Norm(x: &N_Vector, w: &N_Vector) -> sunrealtype {
    let f = x.ops.borrow().nvwl2norm.expect("nvwl2norm");
    f(x, w)
}

pub fn N_VL1Norm(x: &N_Vector) -> sunrealtype {
    let f = x.ops.borrow().nvl1norm.expect("nvl1norm");
    f(x)
}

pub fn N_VCompare(c: sunrealtype, x: &N_Vector, z: &N_Vector) {
    let f = z.ops.borrow().nvcompare.expect("nvcompare");
    f(c, x, z);
}

pub fn N_VInvTest(x: &N_Vector, z: &N_Vector) -> sunbooleantype {
    let f = z.ops.borrow().nvinvtest.expect("nvinvtest");
    f(x, z)
}

pub fn N_VConstrMask(c: &N_Vector, x: &N_Vector, m: &N_Vector) -> sunbooleantype {
    let f = x.ops.borrow().nvconstrmask.expect("nvconstrmask");
    f(c, x, m)
}

pub fn N_VMinQuotient(num: &N_Vector, denom: &N_Vector) -> sunrealtype {
    let f = num.ops.borrow().nvminquotient.expect("nvminquotient");
    f(num, denom)
}

/* -----------------------------------------------------------------
 * OPTIONAL fused vector operations
 * -----------------------------------------------------------------*/

pub fn N_VLinearCombination(
    nvec: i32,
    c: &[sunrealtype],
    X: &[N_Vector],
    z: &N_Vector,
) -> SUNErrCode {
    let f = z.ops.borrow().nvlinearcombination;
    if let Some(f) = f {
        f(nvec, c, X, z)
    } else {
        let nvscale = z.ops.borrow().nvscale.expect("nvscale");
        let nvlinearsum = z.ops.borrow().nvlinearsum.expect("nvlinearsum");
        nvscale(c[0], &X[0], z);
        for i in 1..nvec as usize {
            nvlinearsum(c[i], &X[i], 1.0, z, z);
        }
        SUN_SUCCESS
    }
}

pub fn N_VScaleAddMulti(
    nvec: i32,
    a: &[sunrealtype],
    x: &N_Vector,
    Y: &[N_Vector],
    Z: &[N_Vector],
) -> SUNErrCode {
    let f = x.ops.borrow().nvscaleaddmulti;
    if let Some(f) = f {
        f(nvec, a, x, Y, Z)
    } else {
        let nvlinearsum = x.ops.borrow().nvlinearsum.expect("nvlinearsum");
        for i in 0..nvec as usize {
            nvlinearsum(a[i], x, 1.0, &Y[i], &Z[i]);
        }
        SUN_SUCCESS
    }
}

pub fn N_VDotProdMulti(
    nvec: i32,
    x: &N_Vector,
    Y: &[N_Vector],
    dotprods: &mut [sunrealtype],
) -> SUNErrCode {
    let f = x.ops.borrow().nvdotprodmulti;
    if let Some(f) = f {
        f(nvec, x, Y, dotprods)
    } else {
        let nvdotprod = x.ops.borrow().nvdotprod.expect("nvdotprod");
        for i in 0..nvec as usize {
            dotprods[i] = nvdotprod(x, &Y[i]);
        }
        SUN_SUCCESS
    }
}

/* -----------------------------------------------------------------
 * OPTIONAL vector array operations
 * -----------------------------------------------------------------*/

pub fn N_VLinearSumVectorArray(
    nvec: i32,
    a: sunrealtype,
    X: &[N_Vector],
    b: sunrealtype,
    Y: &[N_Vector],
    Z: &[N_Vector],
) -> SUNErrCode {
    let f = Z[0].ops.borrow().nvlinearsumvectorarray;
    if let Some(f) = f {
        f(nvec, a, X, b, Y, Z)
    } else {
        let nvlinearsum = Z[0].ops.borrow().nvlinearsum.expect("nvlinearsum");
        for i in 0..nvec as usize {
            nvlinearsum(a, &X[i], b, &Y[i], &Z[i]);
        }
        SUN_SUCCESS
    }
}

pub fn N_VScaleVectorArray(
    nvec: i32,
    c: &[sunrealtype],
    X: &[N_Vector],
    Z: &[N_Vector],
) -> SUNErrCode {
    let f = Z[0].ops.borrow().nvscalevectorarray;
    if let Some(f) = f {
        f(nvec, c, X, Z)
    } else {
        let nvscale = Z[0].ops.borrow().nvscale.expect("nvscale");
        for i in 0..nvec as usize {
            nvscale(c[i], &X[i], &Z[i]);
        }
        SUN_SUCCESS
    }
}

pub fn N_VConstVectorArray(nvec: i32, c: sunrealtype, Z: &[N_Vector]) -> SUNErrCode {
    let f = Z[0].ops.borrow().nvconstvectorarray;
    if let Some(f) = f {
        f(nvec, c, Z)
    } else {
        let nvconst = Z[0].ops.borrow().nvconst.expect("nvconst");
        for i in 0..nvec as usize {
            nvconst(c, &Z[i]);
        }
        SUN_SUCCESS
    }
}

pub fn N_VWrmsNormVectorArray(
    nvec: i32,
    X: &[N_Vector],
    W: &[N_Vector],
    nrm: &mut [sunrealtype],
) -> SUNErrCode {
    let f = X[0].ops.borrow().nvwrmsnormvectorarray;
    if let Some(f) = f {
        f(nvec, X, W, nrm)
    } else {
        let nvwrmsnorm = X[0].ops.borrow().nvwrmsnorm.expect("nvwrmsnorm");
        for i in 0..nvec as usize {
            nrm[i] = nvwrmsnorm(&X[i], &W[i]);
        }
        SUN_SUCCESS
    }
}

pub fn N_VWrmsNormMaskVectorArray(
    nvec: i32,
    X: &[N_Vector],
    W: &[N_Vector],
    id: &N_Vector,
    nrm: &mut [sunrealtype],
) -> SUNErrCode {
    let f = id.ops.borrow().nvwrmsnormmaskvectorarray;
    if let Some(f) = f {
        f(nvec, X, W, id, nrm)
    } else {
        let nvwrmsnormmask = id.ops.borrow().nvwrmsnormmask.expect("nvwrmsnormmask");
        for i in 0..nvec as usize {
            nrm[i] = nvwrmsnormmask(&X[i], &W[i], id);
        }
        SUN_SUCCESS
    }
}

pub fn N_VScaleAddMultiVectorArray(
    nvec: i32,
    nsum: i32,
    a: &[sunrealtype],
    X: &[N_Vector],
    Y: &[Vec<N_Vector>],
    Z: &[Vec<N_Vector>],
) -> SUNErrCode {
    let fam = X[0].ops.borrow().nvscaleaddmultivectorarray;
    if let Some(f) = fam {
        return f(nvec, nsum, a, X, Y, Z);
    }
    let fsm = X[0].ops.borrow().nvscaleaddmulti;
    if let Some(f) = fsm {
        let mut ier = SUN_SUCCESS;
        let mut YY: Vec<N_Vector> = Vec::with_capacity(nsum as usize);
        let mut ZZ: Vec<N_Vector> = Vec::with_capacity(nsum as usize);
        for i in 0..nvec as usize {
            YY.clear();
            ZZ.clear();
            for j in 0..nsum as usize {
                YY.push(Y[j][i].clone());
                ZZ.push(Z[j][i].clone());
            }
            ier = f(nsum, a, &X[i], &YY, &ZZ);
            if ier != 0 {
                break;
            }
        }
        ier
    } else {
        let nvlinearsum = X[0].ops.borrow().nvlinearsum.expect("nvlinearsum");
        for i in 0..nvec as usize {
            for j in 0..nsum as usize {
                nvlinearsum(a[j], &X[i], 1.0, &Y[j][i], &Z[j][i]);
            }
        }
        SUN_SUCCESS
    }
}

pub fn N_VLinearCombinationVectorArray(
    nvec: i32,
    nsum: i32,
    c: &[sunrealtype],
    X: &[Vec<N_Vector>],
    Z: &[N_Vector],
) -> SUNErrCode {
    let fca = Z[0].ops.borrow().nvlinearcombinationvectorarray;
    if let Some(f) = fca {
        return f(nvec, nsum, c, X, Z);
    }
    let fc = Z[0].ops.borrow().nvlinearcombination;
    if let Some(f) = fc {
        let mut ier = SUN_SUCCESS;
        let mut Y: Vec<N_Vector> = Vec::with_capacity(nsum as usize);
        for i in 0..nvec as usize {
            Y.clear();
            for j in 0..nsum as usize {
                Y.push(X[j][i].clone());
            }
            ier = f(nsum, c, &Y, &Z[i]);
            if ier != 0 {
                break;
            }
        }
        ier
    } else {
        let nvscale = Z[0].ops.borrow().nvscale.expect("nvscale");
        let nvlinearsum = Z[0].ops.borrow().nvlinearsum.expect("nvlinearsum");
        for i in 0..nvec as usize {
            nvscale(c[0], &X[0][i], &Z[i]);
            for j in 1..nsum as usize {
                nvlinearsum(c[j], &X[j][i], 1.0, &Z[i], &Z[i]);
            }
        }
        SUN_SUCCESS
    }
}

/* -----------------------------------------------------------------
 * OPTIONAL local reduction kernels (no parallel communication)
 * -----------------------------------------------------------------*/

pub fn N_VDotProdLocal(x: &N_Vector, y: &N_Vector) -> sunrealtype {
    let f = y.ops.borrow().nvdotprodlocal.expect("nvdotprodlocal");
    f(x, y)
}

pub fn N_VMaxNormLocal(x: &N_Vector) -> sunrealtype {
    let f = x.ops.borrow().nvmaxnormlocal.expect("nvmaxnormlocal");
    f(x)
}

pub fn N_VMinLocal(x: &N_Vector) -> sunrealtype {
    let f = x.ops.borrow().nvminlocal.expect("nvminlocal");
    f(x)
}

pub fn N_VL1NormLocal(x: &N_Vector) -> sunrealtype {
    let f = x.ops.borrow().nvl1normlocal.expect("nvl1normlocal");
    f(x)
}

pub fn N_VWSqrSumLocal(x: &N_Vector, w: &N_Vector) -> sunrealtype {
    let f = x.ops.borrow().nvwsqrsumlocal.expect("nvwsqrsumlocal");
    f(x, w)
}

pub fn N_VWSqrSumMaskLocal(x: &N_Vector, w: &N_Vector, id: &N_Vector) -> sunrealtype {
    let f = x.ops.borrow().nvwsqrsummasklocal.expect("nvwsqrsummasklocal");
    f(x, w, id)
}

pub fn N_VInvTestLocal(x: &N_Vector, z: &N_Vector) -> sunbooleantype {
    let f = z.ops.borrow().nvinvtestlocal.expect("nvinvtestlocal");
    f(x, z)
}

pub fn N_VConstrMaskLocal(c: &N_Vector, x: &N_Vector, m: &N_Vector) -> sunbooleantype {
    let f = x.ops.borrow().nvconstrmasklocal.expect("nvconstrmasklocal");
    f(c, x, m)
}

pub fn N_VMinQuotientLocal(num: &N_Vector, denom: &N_Vector) -> sunrealtype {
    let f = num.ops.borrow().nvminquotientlocal.expect("nvminquotientlocal");
    f(num, denom)
}

/* -------------------------------------------
 * OPTIONAL single buffer reduction operations
 * -------------------------------------------*/

pub fn N_VDotProdMultiLocal(
    nvec: i32,
    x: &N_Vector,
    Y: &[N_Vector],
    dotprods: &mut [sunrealtype],
) -> SUNErrCode {
    let f = x.ops.borrow().nvdotprodmultilocal;
    if let Some(f) = f {
        return f(nvec, x, Y, dotprods);
    }
    let fl = x.ops.borrow().nvdotprodlocal;
    if let Some(f) = fl {
        for i in 0..nvec as usize {
            dotprods[i] = f(x, &Y[i]);
        }
        return SUN_SUCCESS;
    }
    SUN_SUCCESS
}

pub fn N_VDotProdMultiAllReduce(nvec: i32, x: &N_Vector, sum: &mut [sunrealtype]) -> SUNErrCode {
    let f = x
        .ops
        .borrow()
        .nvdotprodmultiallreduce
        .expect("nvdotprodmultiallreduce");
    f(nvec, x, sum)
}

/* ------------------------------------
 * OPTIONAL XBraid interface operations
 * ------------------------------------*/

pub fn N_VBufSize(x: &N_Vector, size: &mut sunindextype) -> SUNErrCode {
    let f = x.ops.borrow().nvbufsize.expect("nvbufsize");
    f(x, size)
}

pub fn N_VBufPack(x: &N_Vector, buf: &mut [sunrealtype]) -> SUNErrCode {
    let f = x.ops.borrow().nvbufpack.expect("nvbufpack");
    f(x, buf)
}

pub fn N_VBufUnpack(x: &N_Vector, buf: &[sunrealtype]) -> SUNErrCode {
    let f = x.ops.borrow().nvbufunpack.expect("nvbufunpack");
    f(x, buf)
}

/* -----------------------------------------------------------------
 * Additional functions exported by the generic NVECTOR
 * -----------------------------------------------------------------*/

pub fn N_VCloneEmptyVectorArray(count: i32, w: &N_Vector) -> Option<Vec<N_Vector>> {
    let mut vs = Vec::with_capacity(count as usize);
    for _ in 0..count {
        vs.push(N_VCloneEmpty(w)?);
    }
    Some(vs)
}

pub fn N_VCloneVectorArray(count: i32, w: &N_Vector) -> Option<Vec<N_Vector>> {
    let mut vs = Vec::with_capacity(count as usize);
    for _ in 0..count {
        vs.push(N_VClone(w)?);
    }
    Some(vs)
}

pub fn N_VDestroyVectorArray(vs: Vec<N_Vector>, _count: i32) {
    drop(vs);
}

/* -----------------------------------------------------------------
 * Debugging functions
 * ----------------------------------------------------------------- */

pub fn N_VPrint(v: &N_Vector) {
    let f = v.ops.borrow().nvprint;
    match f {
        None => println!("NULL Print Op"),
        Some(f) => f(v),
    }
}

pub fn N_VPrintFile(v: &N_Vector, outfile: &SUNFile) {
    if !outfile.is_null() {
        let f = v.ops.borrow().nvprintfile;
        match f {
            None => outfile.write_str("NULL PrintFile Op\n"),
            Some(f) => f(v, outfile),
        }
    }
}
