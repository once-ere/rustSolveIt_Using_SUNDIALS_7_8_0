//! Port of `src/nvector/serial/nvector_serial.c` +
//! `include/nvector/nvector_serial.h`.
//!
//! Data access: `NV_DATA_S(v)` returns a `RefMut` guard over the data
//! `Vec` (the C `sunrealtype*`); `NV_LENGTH_S(v)` copies out the length.
//! All ops preserve the exact C loop structure and special-case branches
//! (which matter for floating-point arithmetic order). Operand aliasing
//! (C pointer equality) maps to `Rc::ptr_eq`; where C reads through a
//! mutated buffer under aliasing, the Rust code does the same via a
//! single mutable borrow.

use std::cell::RefMut;

use crate::sundials_context::SUNContext;
use crate::sundials_errors::SUN_SUCCESS;
use crate::sundials_math::*;
use crate::sundials_nvector::*;
use crate::sundials_types::*;
use crate::sundials_utils::{sun_format_e, SUNFile};

const ZERO: sunrealtype = 0.0;
const HALF: sunrealtype = 0.5;
const ONE: sunrealtype = 1.0;
const ONEPT5: sunrealtype = 1.5;

pub struct _N_VectorContent_Serial {
    pub length: sunindextype,
    pub own_data: sunbooleantype,
    pub data: Vec<sunrealtype>,
}

pub type N_VectorContent_Serial = _N_VectorContent_Serial;

/// C macro `NV_CONTENT_S(v)` (mutable borrow of the serial content).
fn content_mut(v: &N_Vector) -> RefMut<'_, N_VectorContent_Serial> {
    RefMut::map(v.content.borrow_mut(), |c| {
        c.downcast_mut::<N_VectorContent_Serial>()
            .expect("serial N_Vector content")
    })
}

/// C macro `NV_LENGTH_S(v)`.
pub fn NV_LENGTH_S(v: &N_Vector) -> sunindextype {
    content_mut(v).length
}

/// C macro `NV_OWN_DATA_S(v)`.
pub fn NV_OWN_DATA_S(v: &N_Vector) -> sunbooleantype {
    content_mut(v).own_data
}

/// C macro `NV_DATA_S(v)` — the data pointer as a `RefMut` guard.
/// Drop the guard before calling any other op on the same vector.
pub fn NV_DATA_S(v: &N_Vector) -> RefMut<'_, Vec<sunrealtype>> {
    RefMut::map(v.content.borrow_mut(), |c| {
        &mut c
            .downcast_mut::<N_VectorContent_Serial>()
            .expect("serial N_Vector content")
            .data
    })
}

/// Alias detection: C pointer equality of the vector handles.
fn same(a: &N_Vector, b: &N_Vector) -> bool {
    std::rc::Rc::ptr_eq(a, b)
}

/* ----------------------------------------------------------------------------
 * Function to create a new empty serial vector
 */

pub fn N_VNewEmpty_Serial(length: sunindextype, sunctx: &SUNContext) -> Option<N_Vector> {
    if length < 0 {
        return None;
    }

    /* Create an empty vector object */
    let v = N_VNewEmpty(sunctx)?;

    /* Attach operations */
    {
        let mut ops = v.ops.borrow_mut();

        /* constructors, destructors, and utility operations */
        ops.nvgetvectorid = Some(N_VGetVectorID_Serial);
        ops.nvclone = Some(N_VClone_Serial);
        ops.nvcloneempty = Some(N_VCloneEmpty_Serial);
        ops.nvdestroy = Some(N_VDestroy_Serial);
        ops.nvspace = Some(N_VSpace_Serial);
        ops.nvgetarraypointer = Some(N_VGetArrayPointer_Serial);
        ops.nvsetarraypointer = Some(N_VSetArrayPointer_Serial);
        ops.nvgetlength = Some(N_VGetLength_Serial);
        ops.nvgetlocallength = Some(N_VGetLength_Serial);

        /* standard vector operations */
        ops.nvlinearsum = Some(N_VLinearSum_Serial);
        ops.nvconst = Some(N_VConst_Serial);
        ops.nvprod = Some(N_VProd_Serial);
        ops.nvdiv = Some(N_VDiv_Serial);
        ops.nvscale = Some(N_VScale_Serial);
        ops.nvabs = Some(N_VAbs_Serial);
        ops.nvinv = Some(N_VInv_Serial);
        ops.nvaddconst = Some(N_VAddConst_Serial);
        ops.nvdotprod = Some(N_VDotProd_Serial);
        ops.nvmaxnorm = Some(N_VMaxNorm_Serial);
        ops.nvwrmsnormmask = Some(N_VWrmsNormMask_Serial);
        ops.nvwrmsnorm = Some(N_VWrmsNorm_Serial);
        ops.nvmin = Some(N_VMin_Serial);
        ops.nvwl2norm = Some(N_VWL2Norm_Serial);
        ops.nvl1norm = Some(N_VL1Norm_Serial);
        ops.nvcompare = Some(N_VCompare_Serial);
        ops.nvinvtest = Some(N_VInvTest_Serial);
        ops.nvconstrmask = Some(N_VConstrMask_Serial);
        ops.nvminquotient = Some(N_VMinQuotient_Serial);

        /* fused and vector array operations are disabled (NULL) by default */

        /* local reduction operations */
        ops.nvdotprodlocal = Some(N_VDotProd_Serial);
        ops.nvmaxnormlocal = Some(N_VMaxNorm_Serial);
        ops.nvminlocal = Some(N_VMin_Serial);
        ops.nvl1normlocal = Some(N_VL1Norm_Serial);
        ops.nvinvtestlocal = Some(N_VInvTest_Serial);
        ops.nvconstrmasklocal = Some(N_VConstrMask_Serial);
        ops.nvminquotientlocal = Some(N_VMinQuotient_Serial);
        ops.nvwsqrsumlocal = Some(N_VWSqrSumLocal_Serial);
        ops.nvwsqrsummasklocal = Some(N_VWSqrSumMaskLocal_Serial);

        /* single buffer reduction operations */
        ops.nvdotprodmultilocal = Some(N_VDotProdMulti_Serial);

        /* XBraid interface operations */
        ops.nvbufsize = Some(N_VBufSize_Serial);
        ops.nvbufpack = Some(N_VBufPack_Serial);
        ops.nvbufunpack = Some(N_VBufUnpack_Serial);

        /* debugging functions */
        ops.nvprint = Some(N_VPrint_Serial);
        ops.nvprintfile = Some(N_VPrintFile_Serial);
    }

    /* Create and attach content */
    *v.content.borrow_mut() = Box::new(N_VectorContent_Serial {
        length,
        own_data: SUNFALSE,
        data: Vec::new(),
    });

    Some(v)
}

/* ----------------------------------------------------------------------------
 * Function to create a new serial vector
 */

pub fn N_VNew_Serial(length: sunindextype, sunctx: &SUNContext) -> Option<N_Vector> {
    if length < 0 {
        return None;
    }
    let v = N_VNewEmpty_Serial(length, sunctx)?;

    /* Create and attach data */
    if length > 0 {
        let mut content = content_mut(&v);
        content.own_data = SUNTRUE;
        content.data = vec![0.0; length as usize];
    }

    Some(v)
}

/* ----------------------------------------------------------------------------
 * Function to create a serial N_Vector with user data component
 * (Rust: the vector takes ownership of the provided buffer)
 */

pub fn N_VMake_Serial(
    length: sunindextype,
    v_data: Vec<sunrealtype>,
    sunctx: &SUNContext,
) -> Option<N_Vector> {
    if length < 0 {
        return None;
    }
    let v = N_VNewEmpty_Serial(length, sunctx)?;

    if length > 0 {
        /* Attach data */
        let mut content = content_mut(&v);
        content.own_data = SUNFALSE;
        content.data = v_data;
    }

    Some(v)
}

pub fn N_VGetVectorID_Serial(_v: &N_Vector) -> N_Vector_ID {
    SUNDIALS_NVEC_SERIAL
}

pub fn N_VGetLength_Serial(v: &N_Vector) -> sunindextype {
    NV_LENGTH_S(v)
}

pub fn N_VPrint_Serial(x: &N_Vector) {
    N_VPrintFile_Serial(x, &SUNFile::Stdout);
}

pub fn N_VPrintFile_Serial(x: &N_Vector, outfile: &SUNFile) {
    let n = NV_LENGTH_S(x);
    let xd = NV_DATA_S(x);
    for i in 0..n as usize {
        outfile.write_str(&format!("{}\n", sun_format_e(xd[i])));
    }
}

pub fn N_VCloneEmpty_Serial(w: &N_Vector) -> Option<N_Vector> {
    /* Create vector */
    let v = N_VNewEmpty(&w.sunctx.borrow())?;

    /* Attach operations */
    N_VCopyOps(w, &v);

    /* Create, attach, initialize content */
    *v.content.borrow_mut() = Box::new(N_VectorContent_Serial {
        length: NV_LENGTH_S(w),
        own_data: SUNFALSE,
        data: Vec::new(),
    });

    Some(v)
}

pub fn N_VClone_Serial(w: &N_Vector) -> Option<N_Vector> {
    let v = N_VCloneEmpty_Serial(w)?;

    let length = NV_LENGTH_S(w);

    /* Create data */
    if length > 0 {
        let mut content = content_mut(&v);
        content.own_data = SUNTRUE;
        content.data = vec![0.0; length as usize];
    }

    Some(v)
}

pub fn N_VDestroy_Serial(v: N_Vector) {
    drop(v);
}

pub fn N_VSpace_Serial(v: &N_Vector, lrw: &mut sunindextype, liw: &mut sunindextype) {
    *lrw = NV_LENGTH_S(v);
    *liw = 1;
}

pub fn N_VGetArrayPointer_Serial(v: &N_Vector) -> Option<RefMut<'_, Vec<sunrealtype>>> {
    Some(NV_DATA_S(v))
}

pub fn N_VSetArrayPointer_Serial(v_data: Vec<sunrealtype>, v: &N_Vector) {
    if NV_LENGTH_S(v) > 0 {
        content_mut(v).data = v_data;
    }
}

/* ----------------------------------------------------------------------------
 * Alias-safe elementwise helpers. Each preserves the C loop
 * `for i: z[i] = f(x[i], [y[i]])` exactly; the branches only decide which
 * RefCell borrows are taken so that aliased operands share one borrow.
 */

fn unop(x: &N_Vector, z: &N_Vector, f: impl Fn(sunrealtype) -> sunrealtype) {
    let n = NV_LENGTH_S(x) as usize;
    if same(x, z) {
        let mut zd = NV_DATA_S(z);
        for i in 0..n {
            zd[i] = f(zd[i]);
        }
    } else {
        let xd = NV_DATA_S(x);
        let mut zd = NV_DATA_S(z);
        for i in 0..n {
            zd[i] = f(xd[i]);
        }
    }
}

fn binop(
    x: &N_Vector,
    y: &N_Vector,
    z: &N_Vector,
    f: impl Fn(sunrealtype, sunrealtype) -> sunrealtype,
) {
    let n = NV_LENGTH_S(x) as usize;
    let xz = same(x, z);
    let yz = same(y, z);
    if xz && yz {
        let mut zd = NV_DATA_S(z);
        for i in 0..n {
            zd[i] = f(zd[i], zd[i]);
        }
    } else if xz {
        let yd = NV_DATA_S(y);
        let mut zd = NV_DATA_S(z);
        for i in 0..n {
            zd[i] = f(zd[i], yd[i]);
        }
    } else if yz {
        let xd = NV_DATA_S(x);
        let mut zd = NV_DATA_S(z);
        for i in 0..n {
            zd[i] = f(xd[i], zd[i]);
        }
    } else {
        /* x may alias y: two shared borrows are fine */
        let xd = NV_DATA_S(x);
        let yd = if same(x, y) { None } else { Some(NV_DATA_S(y)) };
        let mut zd = NV_DATA_S(z);
        match yd {
            Some(yd) => {
                for i in 0..n {
                    zd[i] = f(xd[i], yd[i]);
                }
            }
            None => {
                for i in 0..n {
                    zd[i] = f(xd[i], xd[i]);
                }
            }
        }
    }
}

pub fn N_VLinearSum_Serial(
    a: sunrealtype,
    x: &N_Vector,
    b: sunrealtype,
    y: &N_Vector,
    z: &N_Vector,
) {
    if (b == ONE) && same(z, y) {
        /* BLAS usage: axpy y <- ax+y */
        Vaxpy_Serial(a, x, y);
        return;
    }

    if (a == ONE) && same(z, x) {
        /* BLAS usage: axpy x <- by+x */
        Vaxpy_Serial(b, y, x);
        return;
    }

    /* Case: a == b == 1.0 */
    if (a == ONE) && (b == ONE) {
        VSum_Serial(x, y, z);
        return;
    }

    /* Cases: (1) a == 1.0, b = -1.0, (2) a == -1.0, b == 1.0 */
    let test = (a == ONE) && (b == -ONE);
    if test || ((a == -ONE) && (b == ONE)) {
        let v1 = if test { y } else { x };
        let v2 = if test { x } else { y };
        VDiff_Serial(v2, v1, z);
        return;
    }

    /* Cases: (1) a == 1.0, b == other or 0.0, (2) a == other or 0.0, b == 1.0 */
    /* if a or b is 0.0, then user should have called N_VScale */
    let test = a == ONE;
    if test || (b == ONE) {
        let c = if test { b } else { a };
        let v1 = if test { y } else { x };
        let v2 = if test { x } else { y };
        VLin1_Serial(c, v1, v2, z);
        return;
    }

    /* Cases: (1) a == -1.0, b != 1.0, (2) a != 1.0, b == -1.0 */
    let test = a == -ONE;
    if test || (b == -ONE) {
        let c = if test { b } else { a };
        let v1 = if test { y } else { x };
        let v2 = if test { x } else { y };
        VLin2_Serial(c, v1, v2, z);
        return;
    }

    /* Case: a == b (catches a == b == 0.0 - user should have called N_VConst) */
    if a == b {
        VScaleSum_Serial(a, x, y, z);
        return;
    }

    /* Case: a == -b */
    if a == -b {
        VScaleDiff_Serial(a, x, y, z);
        return;
    }

    /* Do all cases not handled above:
    (1) a == other, b == 0.0 - user should have called N_VScale
    (2) a == 0.0, b == other - user should have called N_VScale
    (3) a,b == other, a !=b, a != -b */
    binop(x, y, z, |xi, yi| (a * xi) + (b * yi));
}

pub fn N_VConst_Serial(c: sunrealtype, z: &N_Vector) {
    let n = NV_LENGTH_S(z) as usize;
    let mut zd = NV_DATA_S(z);
    for i in 0..n {
        zd[i] = c;
    }
}

pub fn N_VProd_Serial(x: &N_Vector, y: &N_Vector, z: &N_Vector) {
    binop(x, y, z, |xi, yi| xi * yi);
}

pub fn N_VDiv_Serial(x: &N_Vector, y: &N_Vector, z: &N_Vector) {
    binop(x, y, z, |xi, yi| xi / yi);
}

pub fn N_VScale_Serial(c: sunrealtype, x: &N_Vector, z: &N_Vector) {
    if same(z, x) {
        /* BLAS usage: scale x <- cx */
        VScaleBy_Serial(c, x);
        return;
    }

    if c == ONE {
        VCopy_Serial(x, z);
    } else if c == -ONE {
        VNeg_Serial(x, z);
    } else {
        unop(x, z, |xi| c * xi);
    }
}

pub fn N_VAbs_Serial(x: &N_Vector, z: &N_Vector) {
    unop(x, z, SUNRabs);
}

pub fn N_VInv_Serial(x: &N_Vector, z: &N_Vector) {
    unop(x, z, |xi| ONE / xi);
}

pub fn N_VAddConst_Serial(x: &N_Vector, b: sunrealtype, z: &N_Vector) {
    unop(x, z, |xi| xi + b);
}

pub fn N_VDotProd_Serial(x: &N_Vector, y: &N_Vector) -> sunrealtype {
    let n = NV_LENGTH_S(x) as usize;
    let mut sum = ZERO;
    let xd = NV_DATA_S(x);
    if same(x, y) {
        for i in 0..n {
            sum += xd[i] * xd[i];
        }
    } else {
        let yd = NV_DATA_S(y);
        for i in 0..n {
            sum += xd[i] * yd[i];
        }
    }
    sum
}

pub fn N_VMaxNorm_Serial(x: &N_Vector) -> sunrealtype {
    let n = NV_LENGTH_S(x) as usize;
    let mut max = ZERO;
    let xd = NV_DATA_S(x);
    for i in 0..n {
        if SUNRabs(xd[i]) > max {
            max = SUNRabs(xd[i]);
        }
    }
    max
}

pub fn N_VWrmsNorm_Serial(x: &N_Vector, w: &N_Vector) -> sunrealtype {
    let norm = N_VWSqrSumLocal_Serial(x, w);
    SUNRsqrt(norm / NV_LENGTH_S(x) as sunrealtype)
}

pub fn N_VWSqrSumLocal_Serial(x: &N_Vector, w: &N_Vector) -> sunrealtype {
    let n = NV_LENGTH_S(x) as usize;
    let mut sum = ZERO;
    let xd = NV_DATA_S(x);
    if same(x, w) {
        for i in 0..n {
            let prodi = xd[i] * xd[i];
            sum += SUNSQR(prodi);
        }
    } else {
        let wd = NV_DATA_S(w);
        for i in 0..n {
            let prodi = xd[i] * wd[i];
            sum += SUNSQR(prodi);
        }
    }
    sum
}

pub fn N_VWrmsNormMask_Serial(x: &N_Vector, w: &N_Vector, id: &N_Vector) -> sunrealtype {
    let norm = N_VWSqrSumMaskLocal_Serial(x, w, id);
    SUNRsqrt(norm / NV_LENGTH_S(x) as sunrealtype)
}

pub fn N_VWSqrSumMaskLocal_Serial(x: &N_Vector, w: &N_Vector, id: &N_Vector) -> sunrealtype {
    let n = NV_LENGTH_S(x) as usize;
    let mut sum = ZERO;
    /* all operands are read-only: shared borrows tolerate any aliasing */
    let xd = NV_DATA_S(x);
    let wd = if same(x, w) { None } else { Some(NV_DATA_S(w)) };
    let idd = if same(x, id) {
        None
    } else if same(w, id) && wd.is_some() {
        None
    } else {
        Some(NV_DATA_S(id))
    };
    let get_w = |i: usize| match &wd {
        Some(wd) => wd[i],
        None => xd[i],
    };
    let get_id = |i: usize| {
        if same(x, id) {
            xd[i]
        } else {
            match &idd {
                Some(idd) => idd[i],
                None => get_w(i),
            }
        }
    };
    for i in 0..n {
        if get_id(i) > ZERO {
            let prodi = xd[i] * get_w(i);
            sum += SUNSQR(prodi);
        }
    }
    sum
}

pub fn N_VMin_Serial(x: &N_Vector) -> sunrealtype {
    let n = NV_LENGTH_S(x) as usize;
    let xd = NV_DATA_S(x);
    let mut min = xd[0];
    for i in 1..n {
        if xd[i] < min {
            min = xd[i];
        }
    }
    min
}

pub fn N_VWL2Norm_Serial(x: &N_Vector, w: &N_Vector) -> sunrealtype {
    let n = NV_LENGTH_S(x) as usize;
    let mut sum = ZERO;
    let xd = NV_DATA_S(x);
    if same(x, w) {
        for i in 0..n {
            let prodi = xd[i] * xd[i];
            sum += SUNSQR(prodi);
        }
    } else {
        let wd = NV_DATA_S(w);
        for i in 0..n {
            let prodi = xd[i] * wd[i];
            sum += SUNSQR(prodi);
        }
    }
    SUNRsqrt(sum)
}

pub fn N_VL1Norm_Serial(x: &N_Vector) -> sunrealtype {
    let n = NV_LENGTH_S(x) as usize;
    let mut sum = ZERO;
    let xd = NV_DATA_S(x);
    for i in 0..n {
        sum += SUNRabs(xd[i]);
    }
    sum
}

pub fn N_VCompare_Serial(c: sunrealtype, x: &N_Vector, z: &N_Vector) {
    unop(x, z, |xi| if SUNRabs(xi) >= c { ONE } else { ZERO });
}

pub fn N_VInvTest_Serial(x: &N_Vector, z: &N_Vector) -> sunbooleantype {
    let n = NV_LENGTH_S(x) as usize;
    let mut no_zero_found = SUNTRUE;
    if same(x, z) {
        let mut zd = NV_DATA_S(z);
        for i in 0..n {
            if zd[i] == ZERO {
                no_zero_found = SUNFALSE;
            } else {
                zd[i] = ONE / zd[i];
            }
        }
    } else {
        let xd = NV_DATA_S(x);
        let mut zd = NV_DATA_S(z);
        for i in 0..n {
            if xd[i] == ZERO {
                no_zero_found = SUNFALSE;
            } else {
                zd[i] = ONE / xd[i];
            }
        }
    }
    no_zero_found
}

pub fn N_VConstrMask_Serial(c: &N_Vector, x: &N_Vector, m: &N_Vector) -> sunbooleantype {
    let n = NV_LENGTH_S(x) as usize;
    let mut temp = ZERO;

    /* m never aliases c or x on any in-scope call path; the C code would
    read the freshly-zeroed mask in that case, which the branches below
    reproduce if it ever happens. */
    let mc = same(m, c);
    let mx = same(m, x);
    let cx = same(c, x);

    let mut md = NV_DATA_S(m);
    let cd = if mc { None } else { Some(NV_DATA_S(c)) };
    let xd = if mx {
        None
    } else if cx && cd.is_some() {
        None
    } else {
        Some(NV_DATA_S(x))
    };

    for i in 0..n {
        md[i] = ZERO;

        let cdi = match &cd {
            Some(cd) => cd[i],
            None => md[i],
        };

        /* Continue if no constraints were set for the variable */
        if cdi == ZERO {
            continue;
        }

        let xdi = if mx {
            md[i]
        } else {
            match &xd {
                Some(xd) => xd[i],
                None => cdi,
            }
        };

        /* Check if a set constraint has been violated */
        let test = (SUNRabs(cdi) > ONEPT5 && xdi * cdi <= ZERO)
            || (SUNRabs(cdi) > HALF && xdi * cdi < ZERO);
        if test {
            temp = ONE;
            md[i] = ONE;
        }
    }

    /* Return false if any constraint was violated */
    if temp == ONE {
        SUNFALSE
    } else {
        SUNTRUE
    }
}

pub fn N_VMinQuotient_Serial(num: &N_Vector, denom: &N_Vector) -> sunrealtype {
    let n = NV_LENGTH_S(num) as usize;
    let nd = NV_DATA_S(num);
    let dd = if same(num, denom) {
        None
    } else {
        Some(NV_DATA_S(denom))
    };
    let mut not_even_once = SUNTRUE;
    let mut min = SUN_BIG_REAL;

    for i in 0..n {
        let ddi = match &dd {
            Some(dd) => dd[i],
            None => nd[i],
        };
        if ddi == ZERO {
            continue;
        } else if !not_even_once {
            min = SUNMIN(min, nd[i] / ddi);
        } else {
            min = nd[i] / ddi;
            not_even_once = SUNFALSE;
        }
    }

    min
}

/*
 * -----------------------------------------------------------------
 * fused vector operations
 * -----------------------------------------------------------------
 */

pub fn N_VLinearCombination_Serial(
    nvec: i32,
    c: &[sunrealtype],
    X: &[N_Vector],
    z: &N_Vector,
) -> SUNErrCode {
    /* should have called N_VScale */
    if nvec == 1 {
        N_VScale_Serial(c[0], &X[0], z);
        return SUN_SUCCESS;
    }

    /* should have called N_VLinearSum */
    if nvec == 2 {
        N_VLinearSum_Serial(c[0], &X[0], c[1], &X[1], z);
        return SUN_SUCCESS;
    }

    let n = NV_LENGTH_S(z) as usize;

    /*
     * X[0] += c[i]*X[i], i = 1,...,nvec-1
     */
    if same(&X[0], z) && (c[0] == ONE) {
        let mut zd = NV_DATA_S(z);
        for i in 1..nvec as usize {
            if same(&X[i], z) {
                for j in 0..n {
                    zd[j] += c[i] * zd[j];
                }
            } else {
                let xd = NV_DATA_S(&X[i]);
                for j in 0..n {
                    zd[j] += c[i] * xd[j];
                }
            }
        }
        return SUN_SUCCESS;
    }

    /*
     * X[0] = c[0] * X[0] + sum{ c[i] * X[i] }, i = 1,...,nvec-1
     */
    if same(&X[0], z) {
        let mut zd = NV_DATA_S(z);
        for j in 0..n {
            zd[j] *= c[0];
        }
        for i in 1..nvec as usize {
            if same(&X[i], z) {
                for j in 0..n {
                    zd[j] += c[i] * zd[j];
                }
            } else {
                let xd = NV_DATA_S(&X[i]);
                for j in 0..n {
                    zd[j] += c[i] * xd[j];
                }
            }
        }
        return SUN_SUCCESS;
    }

    /*
     * z = sum{ c[i] * X[i] }, i = 0,...,nvec-1
     */
    {
        let mut zd = NV_DATA_S(z);
        {
            let xd = NV_DATA_S(&X[0]);
            for j in 0..n {
                zd[j] = c[0] * xd[j];
            }
        }
        for i in 1..nvec as usize {
            if same(&X[i], z) {
                for j in 0..n {
                    zd[j] += c[i] * zd[j];
                }
            } else {
                let xd = NV_DATA_S(&X[i]);
                for j in 0..n {
                    zd[j] += c[i] * xd[j];
                }
            }
        }
    }
    SUN_SUCCESS
}

pub fn N_VScaleAddMulti_Serial(
    nvec: i32,
    a: &[sunrealtype],
    x: &N_Vector,
    Y: &[N_Vector],
    Z: &[N_Vector],
) -> SUNErrCode {
    /* should have called N_VLinearSum */
    if nvec == 1 {
        N_VLinearSum_Serial(a[0], x, ONE, &Y[0], &Z[0]);
        return SUN_SUCCESS;
    }

    let n = NV_LENGTH_S(x) as usize;

    /*
     * Y[i][j] += a[i] * x[j]  (C tests array-pointer equality Y == Z)
     */
    if std::ptr::eq(Y.as_ptr(), Z.as_ptr()) {
        let xd = NV_DATA_S(x);
        for i in 0..nvec as usize {
            let mut yd = NV_DATA_S(&Y[i]);
            for j in 0..n {
                yd[j] += a[i] * xd[j];
            }
        }
        return SUN_SUCCESS;
    }

    /*
     * Z[i][j] = Y[i][j] + a[i] * x[j]
     */
    let xd = NV_DATA_S(x);
    for i in 0..nvec as usize {
        if same(&Y[i], &Z[i]) {
            let mut zd = NV_DATA_S(&Z[i]);
            for j in 0..n {
                zd[j] = a[i] * xd[j] + zd[j];
            }
        } else {
            let yd = NV_DATA_S(&Y[i]);
            let mut zd = NV_DATA_S(&Z[i]);
            for j in 0..n {
                zd[j] = a[i] * xd[j] + yd[j];
            }
        }
    }
    SUN_SUCCESS
}

pub fn N_VDotProdMulti_Serial(
    nvec: i32,
    x: &N_Vector,
    Y: &[N_Vector],
    dotprods: &mut [sunrealtype],
) -> SUNErrCode {
    /* should have called N_VDotProd */
    if nvec == 1 {
        dotprods[0] = N_VDotProd_Serial(x, &Y[0]);
        return SUN_SUCCESS;
    }

    let n = NV_LENGTH_S(x) as usize;
    let xd = NV_DATA_S(x);

    /* compute multiple dot products */
    for i in 0..nvec as usize {
        dotprods[i] = ZERO;
        if same(x, &Y[i]) {
            for j in 0..n {
                dotprods[i] += xd[j] * xd[j];
            }
        } else {
            let yd = NV_DATA_S(&Y[i]);
            for j in 0..n {
                dotprods[i] += xd[j] * yd[j];
            }
        }
    }

    SUN_SUCCESS
}

/*
 * -----------------------------------------------------------------
 * vector array operations
 * -----------------------------------------------------------------
 */

/// C array-pointer equality test (`X == Z` on `N_Vector*` arguments).
fn same_array(a: &[N_Vector], b: &[N_Vector]) -> bool {
    std::ptr::eq(a.as_ptr(), b.as_ptr())
}

pub fn N_VLinearSumVectorArray_Serial(
    nvec: i32,
    a: sunrealtype,
    X: &[N_Vector],
    b: sunrealtype,
    Y: &[N_Vector],
    Z: &[N_Vector],
) -> SUNErrCode {
    /* should have called N_VLinearSum */
    if nvec == 1 {
        N_VLinearSum_Serial(a, &X[0], b, &Y[0], &Z[0]);
        return SUN_SUCCESS;
    }

    /* BLAS usage: axpy y <- ax+y */
    if (b == ONE) && same_array(Z, Y) {
        VaxpyVectorArray_Serial(nvec, a, X, Y);
        return SUN_SUCCESS;
    }

    /* BLAS usage: axpy x <- by+x */
    if (a == ONE) && same_array(Z, X) {
        VaxpyVectorArray_Serial(nvec, b, Y, X);
        return SUN_SUCCESS;
    }

    /* Case: a == b == 1.0 */
    if (a == ONE) && (b == ONE) {
        VSumVectorArray_Serial(nvec, X, Y, Z);
        return SUN_SUCCESS;
    }

    /* Cases: (1) a == 1.0, b = -1.0, (2) a == -1.0, b == 1.0 */
    let test = (a == ONE) && (b == -ONE);
    if test || ((a == -ONE) && (b == ONE)) {
        let v1 = if test { Y } else { X };
        let v2 = if test { X } else { Y };
        VDiffVectorArray_Serial(nvec, v2, v1, Z);
        return SUN_SUCCESS;
    }

    /* Cases: (1) a == 1.0, b == other or 0.0, (2) a == other or 0.0, b == 1.0 */
    let test = a == ONE;
    if test || (b == ONE) {
        let c = if test { b } else { a };
        let v1 = if test { Y } else { X };
        let v2 = if test { X } else { Y };
        VLin1VectorArray_Serial(nvec, c, v1, v2, Z);
        return SUN_SUCCESS;
    }

    /* Cases: (1) a == -1.0, b != 1.0, (2) a != 1.0, b == -1.0 */
    let test = a == -ONE;
    if test || (b == -ONE) {
        let c = if test { b } else { a };
        let v1 = if test { Y } else { X };
        let v2 = if test { X } else { Y };
        VLin2VectorArray_Serial(nvec, c, v1, v2, Z);
        return SUN_SUCCESS;
    }

    /* Case: a == b */
    if a == b {
        VScaleSumVectorArray_Serial(nvec, a, X, Y, Z);
        return SUN_SUCCESS;
    }

    /* Case: a == -b */
    if a == -b {
        VScaleDiffVectorArray_Serial(nvec, a, X, Y, Z);
        return SUN_SUCCESS;
    }

    /* compute linear sum for each vector pair in vector arrays */
    for i in 0..nvec as usize {
        binop(&X[i], &Y[i], &Z[i], |xj, yj| a * xj + b * yj);
    }

    SUN_SUCCESS
}

pub fn N_VScaleVectorArray_Serial(
    nvec: i32,
    c: &[sunrealtype],
    X: &[N_Vector],
    Z: &[N_Vector],
) -> SUNErrCode {
    /* should have called N_VScale */
    if nvec == 1 {
        N_VScale_Serial(c[0], &X[0], &Z[0]);
        return SUN_SUCCESS;
    }

    let n = NV_LENGTH_S(&Z[0]) as usize;

    /*
     * X[i] *= c[i]
     */
    if same_array(X, Z) {
        for i in 0..nvec as usize {
            let mut xd = NV_DATA_S(&X[i]);
            for j in 0..n {
                xd[j] *= c[i];
            }
        }
        return SUN_SUCCESS;
    }

    /*
     * Z[i] = c[i] * X[i]
     */
    for i in 0..nvec as usize {
        if same(&X[i], &Z[i]) {
            let mut zd = NV_DATA_S(&Z[i]);
            for j in 0..n {
                zd[j] = c[i] * zd[j];
            }
        } else {
            let xd = NV_DATA_S(&X[i]);
            let mut zd = NV_DATA_S(&Z[i]);
            for j in 0..n {
                zd[j] = c[i] * xd[j];
            }
        }
    }
    SUN_SUCCESS
}

pub fn N_VConstVectorArray_Serial(nvec: i32, c: sunrealtype, Z: &[N_Vector]) -> SUNErrCode {
    /* should have called N_VConst */
    if nvec == 1 {
        N_VConst_Serial(c, &Z[0]);
        return SUN_SUCCESS;
    }

    let n = NV_LENGTH_S(&Z[0]) as usize;

    /* set each vector in the vector array to a constant */
    for i in 0..nvec as usize {
        let mut zd = NV_DATA_S(&Z[i]);
        for j in 0..n {
            zd[j] = c;
        }
    }

    SUN_SUCCESS
}

pub fn N_VWrmsNormVectorArray_Serial(
    nvec: i32,
    X: &[N_Vector],
    W: &[N_Vector],
    nrm: &mut [sunrealtype],
) -> SUNErrCode {
    /* should have called N_VWrmsNorm */
    if nvec == 1 {
        nrm[0] = N_VWrmsNorm_Serial(&X[0], &W[0]);
        return SUN_SUCCESS;
    }

    let n = NV_LENGTH_S(&X[0]) as usize;

    /* compute the WRMS norm for each vector in the vector array */
    for i in 0..nvec as usize {
        let xd = NV_DATA_S(&X[i]);
        let wd = if same(&X[i], &W[i]) {
            None
        } else {
            Some(NV_DATA_S(&W[i]))
        };
        nrm[i] = ZERO;
        match &wd {
            Some(wd) => {
                for j in 0..n {
                    nrm[i] += SUNSQR(xd[j] * wd[j]);
                }
            }
            None => {
                for j in 0..n {
                    nrm[i] += SUNSQR(xd[j] * xd[j]);
                }
            }
        }
        nrm[i] = SUNRsqrt(nrm[i] / n as sunrealtype);
    }

    SUN_SUCCESS
}

pub fn N_VWrmsNormMaskVectorArray_Serial(
    nvec: i32,
    X: &[N_Vector],
    W: &[N_Vector],
    id: &N_Vector,
    nrm: &mut [sunrealtype],
) -> SUNErrCode {
    /* should have called N_VWrmsNorm */
    if nvec == 1 {
        nrm[0] = N_VWrmsNormMask_Serial(&X[0], &W[0], id);
        return SUN_SUCCESS;
    }

    let n = NV_LENGTH_S(&X[0]) as usize;
    let idd = NV_DATA_S(id);

    /* compute the WRMS norm for each vector in the vector array */
    for i in 0..nvec as usize {
        let xd = if same(&X[i], id) {
            None
        } else {
            Some(NV_DATA_S(&X[i]))
        };
        let wd = if same(&W[i], id) || same(&W[i], &X[i]) {
            None
        } else {
            Some(NV_DATA_S(&W[i]))
        };
        nrm[i] = ZERO;
        for j in 0..n {
            let xj = match &xd {
                Some(xd) => xd[j],
                None => idd[j],
            };
            let wj = match &wd {
                Some(wd) => wd[j],
                None => {
                    if same(&W[i], id) {
                        idd[j]
                    } else {
                        xj
                    }
                }
            };
            if idd[j] > ZERO {
                nrm[i] += SUNSQR(xj * wj);
            }
        }
        nrm[i] = SUNRsqrt(nrm[i] / n as sunrealtype);
    }

    SUN_SUCCESS
}

pub fn N_VScaleAddMultiVectorArray_Serial(
    nvec: i32,
    nsum: i32,
    a: &[sunrealtype],
    X: &[N_Vector],
    Y: &[Vec<N_Vector>],
    Z: &[Vec<N_Vector>],
) -> SUNErrCode {
    /* ---------------------------
     * Special cases for nvec == 1
     * --------------------------- */

    if nvec == 1 {
        /* should have called N_VLinearSum */
        if nsum == 1 {
            N_VLinearSum_Serial(a[0], &X[0], ONE, &Y[0][0], &Z[0][0]);
            return SUN_SUCCESS;
        }

        /* should have called N_VScaleAddMulti */
        let YY: Vec<N_Vector> = (0..nsum as usize).map(|j| Y[j][0].clone()).collect();
        let ZZ: Vec<N_Vector> = (0..nsum as usize).map(|j| Z[j][0].clone()).collect();
        let ier = N_VScaleAddMulti_Serial(nsum, a, &X[0], &YY, &ZZ);
        if ier != SUN_SUCCESS {
            return ier;
        }
        return SUN_SUCCESS;
    }

    /* --------------------------
     * Special cases for nvec > 1
     * -------------------------- */

    /* should have called N_VLinearSumVectorArray */
    if nsum == 1 {
        let ier = N_VLinearSumVectorArray_Serial(nvec, a[0], X, ONE, &Y[0], &Z[0]);
        if ier != SUN_SUCCESS {
            return ier;
        }
        return SUN_SUCCESS;
    }

    /* ----------------------------
     * Compute multiple linear sums
     * ---------------------------- */

    let n = NV_LENGTH_S(&X[0]) as usize;

    /*
     * Y[i][j] += a[i] * x[j]  (C tests array-pointer equality Y == Z)
     */
    if std::ptr::eq(Y.as_ptr(), Z.as_ptr()) {
        for i in 0..nvec as usize {
            let xd = NV_DATA_S(&X[i]);
            for j in 0..nsum as usize {
                let mut yd = NV_DATA_S(&Y[j][i]);
                for k in 0..n {
                    yd[k] += a[j] * xd[k];
                }
            }
        }
        return SUN_SUCCESS;
    }

    /*
     * Z[i][j] = Y[i][j] + a[i] * x[j]
     */
    for i in 0..nvec as usize {
        let xd = NV_DATA_S(&X[i]);
        for j in 0..nsum as usize {
            if same(&Y[j][i], &Z[j][i]) {
                let mut zd = NV_DATA_S(&Z[j][i]);
                for k in 0..n {
                    zd[k] = a[j] * xd[k] + zd[k];
                }
            } else {
                let yd = NV_DATA_S(&Y[j][i]);
                let mut zd = NV_DATA_S(&Z[j][i]);
                for k in 0..n {
                    zd[k] = a[j] * xd[k] + yd[k];
                }
            }
        }
    }
    SUN_SUCCESS
}

pub fn N_VLinearCombinationVectorArray_Serial(
    nvec: i32,
    nsum: i32,
    c: &[sunrealtype],
    X: &[Vec<N_Vector>],
    Z: &[N_Vector],
) -> SUNErrCode {
    /* ---------------------------
     * Special cases for nvec == 1
     * --------------------------- */

    if nvec == 1 {
        /* should have called N_VScale */
        if nsum == 1 {
            N_VScale_Serial(c[0], &X[0][0], &Z[0]);
            return SUN_SUCCESS;
        }

        /* should have called N_VLinearSum */
        if nsum == 2 {
            N_VLinearSum_Serial(c[0], &X[0][0], c[1], &X[1][0], &Z[0]);
            return SUN_SUCCESS;
        }

        /* should have called N_VLinearCombination */
        let Y: Vec<N_Vector> = (0..nsum as usize).map(|i| X[i][0].clone()).collect();
        let ier = N_VLinearCombination_Serial(nsum, c, &Y, &Z[0]);
        if ier != SUN_SUCCESS {
            return ier;
        }
        return SUN_SUCCESS;
    }

    /* --------------------------
     * Special cases for nvec > 1
     * -------------------------- */

    /* should have called N_VScaleVectorArray */
    if nsum == 1 {
        let ctmp: Vec<sunrealtype> = vec![c[0]; nvec as usize];
        let ier = N_VScaleVectorArray_Serial(nvec, &ctmp, &X[0], Z);
        if ier != SUN_SUCCESS {
            return ier;
        }
        return SUN_SUCCESS;
    }

    /* should have called N_VLinearSumVectorArray */
    if nsum == 2 {
        let ier = N_VLinearSumVectorArray_Serial(nvec, c[0], &X[0], c[1], &X[1], Z);
        if ier != SUN_SUCCESS {
            return ier;
        }
        return SUN_SUCCESS;
    }

    /* --------------------------
     * Compute linear combination
     * -------------------------- */

    let n = NV_LENGTH_S(&Z[0]) as usize;

    /*
     * X[0][j] += c[i]*X[i][j], i = 1,...,nvec-1
     */
    if same_array(&X[0], Z) && (c[0] == ONE) {
        for j in 0..nvec as usize {
            let mut zd = NV_DATA_S(&Z[j]);
            for i in 1..nsum as usize {
                if same(&X[i][j], &Z[j]) {
                    for k in 0..n {
                        zd[k] += c[i] * zd[k];
                    }
                } else {
                    let xd = NV_DATA_S(&X[i][j]);
                    for k in 0..n {
                        zd[k] += c[i] * xd[k];
                    }
                }
            }
        }
        return SUN_SUCCESS;
    }

    /*
     * X[0][j] = c[0] * X[0][j] + sum{ c[i] * X[i][j] }, i = 1,...,nvec-1
     */
    if same_array(&X[0], Z) {
        for j in 0..nvec as usize {
            let mut zd = NV_DATA_S(&Z[j]);
            for k in 0..n {
                zd[k] *= c[0];
            }
            for i in 1..nsum as usize {
                if same(&X[i][j], &Z[j]) {
                    for k in 0..n {
                        zd[k] += c[i] * zd[k];
                    }
                } else {
                    let xd = NV_DATA_S(&X[i][j]);
                    for k in 0..n {
                        zd[k] += c[i] * xd[k];
                    }
                }
            }
        }
        return SUN_SUCCESS;
    }

    /*
     * Z[j] = sum{ c[i] * X[i][j] }, i = 0,...,nvec-1
     */
    for j in 0..nvec as usize {
        {
            if same(&X[0][j], &Z[j]) {
                let mut zd = NV_DATA_S(&Z[j]);
                for k in 0..n {
                    zd[k] = c[0] * zd[k];
                }
            } else {
                let xd = NV_DATA_S(&X[0][j]);
                let mut zd = NV_DATA_S(&Z[j]);
                for k in 0..n {
                    zd[k] = c[0] * xd[k];
                }
            }
        }
        let mut zd = NV_DATA_S(&Z[j]);
        for i in 1..nsum as usize {
            if same(&X[i][j], &Z[j]) {
                for k in 0..n {
                    zd[k] += c[i] * zd[k];
                }
            } else {
                let xd = NV_DATA_S(&X[i][j]);
                for k in 0..n {
                    zd[k] += c[i] * xd[k];
                }
            }
        }
    }
    SUN_SUCCESS
}

/*
 * -----------------------------------------------------------------
 * OPTIONAL XBraid interface operations
 * -----------------------------------------------------------------
 */

pub fn N_VBufSize_Serial(x: &N_Vector, size: &mut sunindextype) -> SUNErrCode {
    *size = NV_LENGTH_S(x) * (std::mem::size_of::<sunrealtype>() as sunindextype);
    SUN_SUCCESS
}

pub fn N_VBufPack_Serial(x: &N_Vector, buf: &mut [sunrealtype]) -> SUNErrCode {
    let n = NV_LENGTH_S(x) as usize;
    let xd = NV_DATA_S(x);
    buf[..n].copy_from_slice(&xd[..n]);
    SUN_SUCCESS
}

pub fn N_VBufUnpack_Serial(x: &N_Vector, buf: &[sunrealtype]) -> SUNErrCode {
    let n = NV_LENGTH_S(x) as usize;
    let mut xd = NV_DATA_S(x);
    xd[..n].copy_from_slice(&buf[..n]);
    SUN_SUCCESS
}

/*
 * -----------------------------------------------------------------
 * private functions for special cases of vector operations
 * -----------------------------------------------------------------
 */

fn VCopy_Serial(x: &N_Vector, z: &N_Vector) {
    unop(x, z, |xi| xi);
}

fn VSum_Serial(x: &N_Vector, y: &N_Vector, z: &N_Vector) {
    binop(x, y, z, |xi, yi| xi + yi);
}

fn VDiff_Serial(x: &N_Vector, y: &N_Vector, z: &N_Vector) {
    binop(x, y, z, |xi, yi| xi - yi);
}

fn VNeg_Serial(x: &N_Vector, z: &N_Vector) {
    unop(x, z, |xi| -xi);
}

fn VScaleSum_Serial(c: sunrealtype, x: &N_Vector, y: &N_Vector, z: &N_Vector) {
    binop(x, y, z, |xi, yi| c * (xi + yi));
}

fn VScaleDiff_Serial(c: sunrealtype, x: &N_Vector, y: &N_Vector, z: &N_Vector) {
    binop(x, y, z, |xi, yi| c * (xi - yi));
}

fn VLin1_Serial(a: sunrealtype, x: &N_Vector, y: &N_Vector, z: &N_Vector) {
    binop(x, y, z, |xi, yi| (a * xi) + yi);
}

fn VLin2_Serial(a: sunrealtype, x: &N_Vector, y: &N_Vector, z: &N_Vector) {
    binop(x, y, z, |xi, yi| (a * xi) - yi);
}

fn Vaxpy_Serial(a: sunrealtype, x: &N_Vector, y: &N_Vector) {
    let n = NV_LENGTH_S(x) as usize;
    if same(x, y) {
        let mut yd = NV_DATA_S(y);
        if a == ONE {
            for i in 0..n {
                let xi = yd[i];
                yd[i] += xi;
            }
            return;
        }
        if a == -ONE {
            for i in 0..n {
                let xi = yd[i];
                yd[i] -= xi;
            }
            return;
        }
        for i in 0..n {
            let xi = yd[i];
            yd[i] += a * xi;
        }
        return;
    }

    let xd = NV_DATA_S(x);
    let mut yd = NV_DATA_S(y);

    if a == ONE {
        for i in 0..n {
            yd[i] += xd[i];
        }
        return;
    }

    if a == -ONE {
        for i in 0..n {
            yd[i] -= xd[i];
        }
        return;
    }

    for i in 0..n {
        yd[i] += a * xd[i];
    }
}

fn VScaleBy_Serial(a: sunrealtype, x: &N_Vector) {
    let n = NV_LENGTH_S(x) as usize;
    let mut xd = NV_DATA_S(x);
    for i in 0..n {
        xd[i] *= a;
    }
}

/*
 * -----------------------------------------------------------------
 * private functions for special cases of vector array operations
 * -----------------------------------------------------------------
 */

fn VSumVectorArray_Serial(nvec: i32, X: &[N_Vector], Y: &[N_Vector], Z: &[N_Vector]) {
    for i in 0..nvec as usize {
        binop(&X[i], &Y[i], &Z[i], |xj, yj| xj + yj);
    }
}

fn VDiffVectorArray_Serial(nvec: i32, X: &[N_Vector], Y: &[N_Vector], Z: &[N_Vector]) {
    for i in 0..nvec as usize {
        binop(&X[i], &Y[i], &Z[i], |xj, yj| xj - yj);
    }
}

fn VScaleSumVectorArray_Serial(
    nvec: i32,
    c: sunrealtype,
    X: &[N_Vector],
    Y: &[N_Vector],
    Z: &[N_Vector],
) {
    for i in 0..nvec as usize {
        binop(&X[i], &Y[i], &Z[i], |xj, yj| c * (xj + yj));
    }
}

fn VScaleDiffVectorArray_Serial(
    nvec: i32,
    c: sunrealtype,
    X: &[N_Vector],
    Y: &[N_Vector],
    Z: &[N_Vector],
) {
    for i in 0..nvec as usize {
        binop(&X[i], &Y[i], &Z[i], |xj, yj| c * (xj - yj));
    }
}

fn VLin1VectorArray_Serial(
    nvec: i32,
    a: sunrealtype,
    X: &[N_Vector],
    Y: &[N_Vector],
    Z: &[N_Vector],
) {
    for i in 0..nvec as usize {
        binop(&X[i], &Y[i], &Z[i], |xj, yj| (a * xj) + yj);
    }
}

fn VLin2VectorArray_Serial(
    nvec: i32,
    a: sunrealtype,
    X: &[N_Vector],
    Y: &[N_Vector],
    Z: &[N_Vector],
) {
    for i in 0..nvec as usize {
        binop(&X[i], &Y[i], &Z[i], |xj, yj| (a * xj) - yj);
    }
}

fn VaxpyVectorArray_Serial(nvec: i32, a: sunrealtype, X: &[N_Vector], Y: &[N_Vector]) {
    for i in 0..nvec as usize {
        Vaxpy_Serial(a, &X[i], &Y[i]);
    }
}

/*
 * -----------------------------------------------------------------
 * Enable / Disable fused and vector array operations
 * -----------------------------------------------------------------
 */

pub fn N_VEnableFusedOps_Serial(v: &N_Vector, tf: sunbooleantype) -> SUNErrCode {
    let mut ops = v.ops.borrow_mut();
    if tf {
        /* enable all fused vector operations */
        ops.nvlinearcombination = Some(N_VLinearCombination_Serial);
        ops.nvscaleaddmulti = Some(N_VScaleAddMulti_Serial);
        ops.nvdotprodmulti = Some(N_VDotProdMulti_Serial);
        /* enable all vector array operations */
        ops.nvlinearsumvectorarray = Some(N_VLinearSumVectorArray_Serial);
        ops.nvscalevectorarray = Some(N_VScaleVectorArray_Serial);
        ops.nvconstvectorarray = Some(N_VConstVectorArray_Serial);
        ops.nvwrmsnormvectorarray = Some(N_VWrmsNormVectorArray_Serial);
        ops.nvwrmsnormmaskvectorarray = Some(N_VWrmsNormMaskVectorArray_Serial);
        ops.nvscaleaddmultivectorarray = Some(N_VScaleAddMultiVectorArray_Serial);
        ops.nvlinearcombinationvectorarray = Some(N_VLinearCombinationVectorArray_Serial);
        /* enable single buffer reduction operations */
        ops.nvdotprodmultilocal = Some(N_VDotProdMulti_Serial);
    } else {
        /* disable all fused vector operations */
        ops.nvlinearcombination = None;
        ops.nvscaleaddmulti = None;
        ops.nvdotprodmulti = None;
        /* disable all vector array operations */
        ops.nvlinearsumvectorarray = None;
        ops.nvscalevectorarray = None;
        ops.nvconstvectorarray = None;
        ops.nvwrmsnormvectorarray = None;
        ops.nvwrmsnormmaskvectorarray = None;
        ops.nvscaleaddmultivectorarray = None;
        ops.nvlinearcombinationvectorarray = None;
        /* disable single buffer reduction operations */
        ops.nvdotprodmultilocal = None;
    }
    SUN_SUCCESS
}

pub fn N_VEnableLinearCombination_Serial(v: &N_Vector, tf: sunbooleantype) -> SUNErrCode {
    v.ops.borrow_mut().nvlinearcombination = if tf {
        Some(N_VLinearCombination_Serial)
    } else {
        None
    };
    SUN_SUCCESS
}

pub fn N_VEnableScaleAddMulti_Serial(v: &N_Vector, tf: sunbooleantype) -> SUNErrCode {
    v.ops.borrow_mut().nvscaleaddmulti = if tf { Some(N_VScaleAddMulti_Serial) } else { None };
    SUN_SUCCESS
}

pub fn N_VEnableDotProdMulti_Serial(v: &N_Vector, tf: sunbooleantype) -> SUNErrCode {
    let mut ops = v.ops.borrow_mut();
    ops.nvdotprodmulti = if tf { Some(N_VDotProdMulti_Serial) } else { None };
    ops.nvdotprodmultilocal = if tf { Some(N_VDotProdMulti_Serial) } else { None };
    SUN_SUCCESS
}

pub fn N_VEnableLinearSumVectorArray_Serial(v: &N_Vector, tf: sunbooleantype) -> SUNErrCode {
    v.ops.borrow_mut().nvlinearsumvectorarray = if tf {
        Some(N_VLinearSumVectorArray_Serial)
    } else {
        None
    };
    SUN_SUCCESS
}

pub fn N_VEnableScaleVectorArray_Serial(v: &N_Vector, tf: sunbooleantype) -> SUNErrCode {
    v.ops.borrow_mut().nvscalevectorarray = if tf {
        Some(N_VScaleVectorArray_Serial)
    } else {
        None
    };
    SUN_SUCCESS
}

pub fn N_VEnableConstVectorArray_Serial(v: &N_Vector, tf: sunbooleantype) -> SUNErrCode {
    v.ops.borrow_mut().nvconstvectorarray = if tf {
        Some(N_VConstVectorArray_Serial)
    } else {
        None
    };
    SUN_SUCCESS
}

pub fn N_VEnableWrmsNormVectorArray_Serial(v: &N_Vector, tf: sunbooleantype) -> SUNErrCode {
    v.ops.borrow_mut().nvwrmsnormvectorarray = if tf {
        Some(N_VWrmsNormVectorArray_Serial)
    } else {
        None
    };
    SUN_SUCCESS
}

pub fn N_VEnableWrmsNormMaskVectorArray_Serial(v: &N_Vector, tf: sunbooleantype) -> SUNErrCode {
    v.ops.borrow_mut().nvwrmsnormmaskvectorarray = if tf {
        Some(N_VWrmsNormMaskVectorArray_Serial)
    } else {
        None
    };
    SUN_SUCCESS
}

pub fn N_VEnableScaleAddMultiVectorArray_Serial(v: &N_Vector, tf: sunbooleantype) -> SUNErrCode {
    v.ops.borrow_mut().nvscaleaddmultivectorarray = if tf {
        Some(N_VScaleAddMultiVectorArray_Serial)
    } else {
        None
    };
    SUN_SUCCESS
}

pub fn N_VEnableLinearCombinationVectorArray_Serial(
    v: &N_Vector,
    tf: sunbooleantype,
) -> SUNErrCode {
    v.ops.borrow_mut().nvlinearcombinationvectorarray = if tf {
        Some(N_VLinearCombinationVectorArray_Serial)
    } else {
        None
    };
    SUN_SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sundials_context::{SUNContext_Create, SUNContext};

    fn ctx() -> SUNContext {
        let mut c = None;
        assert_eq!(SUNContext_Create(0, &mut c), SUN_SUCCESS);
        c.expect("context created")
    }

    #[test]
    fn create_fill_ops() {
        let sunctx = ctx();
        let x = N_VNew_Serial(4, &sunctx).expect("vector");
        assert_eq!(N_VGetLength(&x), 4);
        N_VConst(2.0, &x);
        let y = N_VClone(&x).expect("clone");
        N_VConst(3.0, &y);
        let z = N_VClone(&x).expect("clone");
        N_VLinearSum(1.0, &x, 1.0, &y, &z);
        assert_eq!(NV_DATA_S(&z)[0], 5.0);
        /* aliased: z = 2*z - x */
        N_VLinearSum(2.0, &z, -1.0, &x, &z);
        assert_eq!(NV_DATA_S(&z)[1], 8.0);
        assert_eq!(N_VDotProd(&x, &y), 24.0);
        assert_eq!(N_VMaxNorm(&z), 8.0);
        assert_eq!(N_VMin(&x), 2.0);
        /* wrms of constant vector v with weight w: |v*w| */
        let w = N_VClone(&x).expect("clone");
        N_VConst(0.5, &w);
        assert_eq!(N_VWrmsNorm(&x, &w), 1.0);
    }

    #[test]
    fn fused_and_arrays() {
        let sunctx = ctx();
        let x = N_VNew_Serial(3, &sunctx).expect("vector");
        N_VEnableFusedOps_Serial(&x, SUNTRUE);
        N_VConst(1.0, &x);
        let ys = N_VCloneVectorArray(3, &x).expect("array");
        for (i, y) in ys.iter().enumerate() {
            N_VConst((i + 1) as sunrealtype, y);
        }
        let z = N_VClone(&x).expect("clone");
        let c = [1.0, 2.0, 3.0];
        assert_eq!(N_VLinearCombination(3, &c, &ys, &z), SUN_SUCCESS);
        /* 1*1 + 2*2 + 3*3 = 14 */
        assert_eq!(NV_DATA_S(&z)[0], 14.0);
    }
}
