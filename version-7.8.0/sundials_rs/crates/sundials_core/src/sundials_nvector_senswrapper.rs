//! Port of `src/sundials/sundials_nvector_senswrapper.c` +
//! `include/sundials/sundials_nvector_senswrapper.h` (vector-of-vectors
//! wrapper used by the CVODES/IDAS sensitivity nonlinear solvers).

use std::cell::RefMut;

use crate::sundials_context::SUNContext;
use crate::sundials_nvector::*;
use crate::sundials_types::*;

const ZERO: sunrealtype = 0.0;

pub struct _N_VectorContent_SensWrapper {
    pub nvecs: i32,
    pub own_vecs: sunbooleantype,
    pub vecs: Vec<Option<N_Vector>>,
}

pub type N_VectorContent_SensWrapper = _N_VectorContent_SensWrapper;

fn content_mut(v: &N_Vector) -> RefMut<'_, _N_VectorContent_SensWrapper> {
    RefMut::map(v.content.borrow_mut(), |c| {
        c.downcast_mut::<_N_VectorContent_SensWrapper>()
            .expect("SensWrapper N_Vector content")
    })
}

/// C macro `NV_NVECS_SW(v)`.
pub fn NV_NVECS_SW(v: &N_Vector) -> i32 {
    content_mut(v).nvecs
}

/// C macro `NV_OWN_VECS_SW(v)`.
pub fn NV_OWN_VECS_SW(v: &N_Vector) -> sunbooleantype {
    content_mut(v).own_vecs
}

/// C macro `NV_VEC_SW(v, i)` (read).
pub fn NV_VEC_SW(v: &N_Vector, i: i32) -> N_Vector {
    content_mut(v).vecs[i as usize]
        .as_ref()
        .expect("SensWrapper subvector")
        .clone()
}

/// C macro `NV_VEC_SW(v, i) = w` (write).
pub fn NV_VEC_SW_set(v: &N_Vector, i: i32, w: Option<N_Vector>) {
    content_mut(v).vecs[i as usize] = w;
}

pub fn N_VNewEmpty_SensWrapper(nvecs: i32, sunctx: &SUNContext) -> Option<N_Vector> {
    /* return if wrapper is empty */
    if nvecs < 1 {
        return None;
    }

    let v = N_VNewEmpty(sunctx)?;

    /* Attach operations */
    {
        let mut ops = v.ops.borrow_mut();
        ops.nvclone = Some(N_VClone_SensWrapper);
        ops.nvcloneempty = Some(N_VCloneEmpty_SensWrapper);
        ops.nvdestroy = Some(N_VDestroy_SensWrapper);

        /* standard vector operations */
        ops.nvlinearsum = Some(N_VLinearSum_SensWrapper);
        ops.nvconst = Some(N_VConst_SensWrapper);
        ops.nvprod = Some(N_VProd_SensWrapper);
        ops.nvdiv = Some(N_VDiv_SensWrapper);
        ops.nvscale = Some(N_VScale_SensWrapper);
        ops.nvabs = Some(N_VAbs_SensWrapper);
        ops.nvinv = Some(N_VInv_SensWrapper);
        ops.nvaddconst = Some(N_VAddConst_SensWrapper);
        ops.nvdotprod = Some(N_VDotProd_SensWrapper);
        ops.nvmaxnorm = Some(N_VMaxNorm_SensWrapper);
        ops.nvwrmsnormmask = Some(N_VWrmsNormMask_SensWrapper);
        ops.nvwrmsnorm = Some(N_VWrmsNorm_SensWrapper);
        ops.nvmin = Some(N_VMin_SensWrapper);
        ops.nvwl2norm = Some(N_VWL2Norm_SensWrapper);
        ops.nvl1norm = Some(N_VL1Norm_SensWrapper);
        ops.nvcompare = Some(N_VCompare_SensWrapper);
        ops.nvinvtest = Some(N_VInvTest_SensWrapper);
        ops.nvconstrmask = Some(N_VConstrMask_SensWrapper);
        ops.nvminquotient = Some(N_VMinQuotient_SensWrapper);
    }

    /* create and attach content */
    *v.content.borrow_mut() = Box::new(_N_VectorContent_SensWrapper {
        nvecs,
        own_vecs: SUNFALSE,
        vecs: (0..nvecs).map(|_| None).collect(),
    });

    Some(v)
}

pub fn N_VNew_SensWrapper(count: i32, w: &N_Vector) -> Option<N_Vector> {
    let v = N_VNewEmpty_SensWrapper(count, &w.sunctx.borrow())?;

    for i in 0..NV_NVECS_SW(&v) {
        let sub = N_VClone(w)?;
        NV_VEC_SW_set(&v, i, Some(sub));
    }

    /* update own vectors status */
    content_mut(&v).own_vecs = SUNTRUE;

    /* set context */
    *v.sunctx.borrow_mut() = w.sunctx.borrow().clone();

    Some(v)
}

pub fn N_VCloneEmpty_SensWrapper(w: &N_Vector) -> Option<N_Vector> {
    if NV_NVECS_SW(w) < 1 {
        return None;
    }

    /* create vector with w's full ops table */
    let v = N_VNewEmpty(&w.sunctx.borrow())?;
    N_VCopyOps(w, &v);

    let nvecs = NV_NVECS_SW(w);
    *v.content.borrow_mut() = Box::new(_N_VectorContent_SensWrapper {
        nvecs,
        own_vecs: SUNFALSE,
        vecs: (0..nvecs).map(|_| None).collect(),
    });

    Some(v)
}

pub fn N_VClone_SensWrapper(w: &N_Vector) -> Option<N_Vector> {
    /* create empty wrapper */
    let v = N_VCloneEmpty_SensWrapper(w)?;

    /* update own vectors status */
    content_mut(&v).own_vecs = SUNTRUE;

    /* allocate arrays */
    for i in 0..NV_NVECS_SW(&v) {
        let sub = N_VClone(&NV_VEC_SW(w, i))?;
        NV_VEC_SW_set(&v, i, Some(sub));
    }

    Some(v)
}

pub fn N_VDestroy_SensWrapper(v: N_Vector) {
    drop(v);
}

pub fn N_VLinearSum_SensWrapper(
    a: sunrealtype,
    x: &N_Vector,
    b: sunrealtype,
    y: &N_Vector,
    z: &N_Vector,
) {
    for i in 0..NV_NVECS_SW(x) {
        N_VLinearSum(a, &NV_VEC_SW(x, i), b, &NV_VEC_SW(y, i), &NV_VEC_SW(z, i));
    }
}

pub fn N_VConst_SensWrapper(c: sunrealtype, z: &N_Vector) {
    for i in 0..NV_NVECS_SW(z) {
        N_VConst(c, &NV_VEC_SW(z, i));
    }
}

pub fn N_VProd_SensWrapper(x: &N_Vector, y: &N_Vector, z: &N_Vector) {
    for i in 0..NV_NVECS_SW(x) {
        N_VProd(&NV_VEC_SW(x, i), &NV_VEC_SW(y, i), &NV_VEC_SW(z, i));
    }
}

pub fn N_VDiv_SensWrapper(x: &N_Vector, y: &N_Vector, z: &N_Vector) {
    for i in 0..NV_NVECS_SW(x) {
        N_VDiv(&NV_VEC_SW(x, i), &NV_VEC_SW(y, i), &NV_VEC_SW(z, i));
    }
}

pub fn N_VScale_SensWrapper(c: sunrealtype, x: &N_Vector, z: &N_Vector) {
    for i in 0..NV_NVECS_SW(x) {
        N_VScale(c, &NV_VEC_SW(x, i), &NV_VEC_SW(z, i));
    }
}

pub fn N_VAbs_SensWrapper(x: &N_Vector, z: &N_Vector) {
    for i in 0..NV_NVECS_SW(x) {
        N_VAbs(&NV_VEC_SW(x, i), &NV_VEC_SW(z, i));
    }
}

pub fn N_VInv_SensWrapper(x: &N_Vector, z: &N_Vector) {
    for i in 0..NV_NVECS_SW(x) {
        N_VInv(&NV_VEC_SW(x, i), &NV_VEC_SW(z, i));
    }
}

pub fn N_VAddConst_SensWrapper(x: &N_Vector, b: sunrealtype, z: &N_Vector) {
    for i in 0..NV_NVECS_SW(x) {
        N_VAddConst(&NV_VEC_SW(x, i), b, &NV_VEC_SW(z, i));
    }
}

pub fn N_VDotProd_SensWrapper(x: &N_Vector, y: &N_Vector) -> sunrealtype {
    let mut sum = ZERO;
    for i in 0..NV_NVECS_SW(x) {
        sum += N_VDotProd(&NV_VEC_SW(x, i), &NV_VEC_SW(y, i));
    }
    sum
}

pub fn N_VMaxNorm_SensWrapper(x: &N_Vector) -> sunrealtype {
    let mut max = ZERO;
    for i in 0..NV_NVECS_SW(x) {
        let tmp = N_VMaxNorm(&NV_VEC_SW(x, i));
        if tmp > max {
            max = tmp;
        }
    }
    max
}

pub fn N_VWrmsNorm_SensWrapper(x: &N_Vector, w: &N_Vector) -> sunrealtype {
    let mut nrm = ZERO;
    for i in 0..NV_NVECS_SW(x) {
        let tmp = N_VWrmsNorm(&NV_VEC_SW(x, i), &NV_VEC_SW(w, i));
        if tmp > nrm {
            nrm = tmp;
        }
    }
    nrm
}

pub fn N_VWrmsNormMask_SensWrapper(x: &N_Vector, w: &N_Vector, id: &N_Vector) -> sunrealtype {
    let mut nrm = ZERO;
    for i in 0..NV_NVECS_SW(x) {
        let tmp = N_VWrmsNormMask(&NV_VEC_SW(x, i), &NV_VEC_SW(w, i), &NV_VEC_SW(id, i));
        if tmp > nrm {
            nrm = tmp;
        }
    }
    nrm
}

pub fn N_VMin_SensWrapper(x: &N_Vector) -> sunrealtype {
    let mut min = N_VMin(&NV_VEC_SW(x, 0));
    for i in 1..NV_NVECS_SW(x) {
        let tmp = N_VMin(&NV_VEC_SW(x, i));
        if tmp < min {
            min = tmp;
        }
    }
    min
}

pub fn N_VWL2Norm_SensWrapper(x: &N_Vector, w: &N_Vector) -> sunrealtype {
    let mut nrm = ZERO;
    for i in 0..NV_NVECS_SW(x) {
        let tmp = N_VWL2Norm(&NV_VEC_SW(x, i), &NV_VEC_SW(w, i));
        if tmp > nrm {
            nrm = tmp;
        }
    }
    nrm
}

pub fn N_VL1Norm_SensWrapper(x: &N_Vector) -> sunrealtype {
    let mut nrm = ZERO;
    for i in 0..NV_NVECS_SW(x) {
        let tmp = N_VL1Norm(&NV_VEC_SW(x, i));
        if tmp > nrm {
            nrm = tmp;
        }
    }
    nrm
}

pub fn N_VCompare_SensWrapper(c: sunrealtype, x: &N_Vector, z: &N_Vector) {
    for i in 0..NV_NVECS_SW(x) {
        N_VCompare(c, &NV_VEC_SW(x, i), &NV_VEC_SW(z, i));
    }
}

pub fn N_VInvTest_SensWrapper(x: &N_Vector, z: &N_Vector) -> sunbooleantype {
    let mut no_zero_found = SUNTRUE;
    for i in 0..NV_NVECS_SW(x) {
        let tmp = N_VInvTest(&NV_VEC_SW(x, i), &NV_VEC_SW(z, i));
        if tmp != SUNTRUE {
            no_zero_found = SUNFALSE;
        }
    }
    no_zero_found
}

/// Note: `c` is the shared (non-wrapper) constraints vector, passed through
/// unchanged to each subvector test exactly as upstream does.
pub fn N_VConstrMask_SensWrapper(c: &N_Vector, x: &N_Vector, m: &N_Vector) -> sunbooleantype {
    let mut test = SUNTRUE;
    for i in 0..NV_NVECS_SW(x) {
        let tmp = N_VConstrMask(c, &NV_VEC_SW(x, i), &NV_VEC_SW(m, i));
        if tmp != SUNTRUE {
            test = SUNFALSE;
        }
    }
    test
}

pub fn N_VMinQuotient_SensWrapper(num: &N_Vector, denom: &N_Vector) -> sunrealtype {
    let mut min = N_VMinQuotient(&NV_VEC_SW(num, 0), &NV_VEC_SW(denom, 0));
    for i in 1..NV_NVECS_SW(num) {
        let tmp = N_VMinQuotient(&NV_VEC_SW(num, i), &NV_VEC_SW(denom, i));
        if tmp < min {
            min = tmp;
        }
    }
    min
}
