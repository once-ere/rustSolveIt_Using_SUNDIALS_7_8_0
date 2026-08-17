//! Port of `src/nvector/manyvector/nvector_manyvector.c` +
//! `include/nvector/nvector_manyvector.h` — **serial variant only**.
//!
//! The upstream C file builds both the serial ManyVector and the MPI
//! ManyVector from one source, switched by `MANYVECTOR_BUILD_WITH_MPI`.
//! This module is the `#else` (non-MPI) branch throughout: `MVAPPEND(fun)`
//! is `fun##_ManyVector`, the communicator field and every MPI-only entry
//! point (`N_VMake_MPIManyVector`, `N_VNew_MPIManyVector`,
//! `N_VGetCommunicator_MPIManyVector`, the `MPI_Allreduce` combiners,
//! `N_VDotProdMultiAllReduce_MPIManyVector`, `SubvectorMPIRank`) are absent.
//!
//! Handle model (ARCHITECTURE.md): the C `N_Vector* subvec_array` is a
//! `Vec<N_Vector>` of `Rc` handles — cloning a handle *is* the C pointer
//! copy, so `MANYVECTOR_SUBVEC` hands out a clone and no content borrow is
//! ever held across an operation on a subvector. Every subvector-iterating
//! op delegates through the generic `N_V*` free functions in the C loop
//! order, and reductions accumulate subvector contributions in C order —
//! floating-point non-associativity is output-observable.

use std::cell::RefMut;

use crate::sundials_context::SUNContext;
use crate::sundials_errors::SUN_SUCCESS;
use crate::sundials_math::SUNRsqrt;
use crate::sundials_nvector::*;
use crate::sundials_types::*;
use crate::sundials_utils::SUNFile;

const ZERO: sunrealtype = 0.0;

/* -----------------------------------------------------------------
   ManyVector implementation of N_Vector
   ----------------------------------------------------------------- */

pub struct _N_VectorContent_ManyVector {
    /// number of vectors attached
    pub num_subvectors: sunindextype,
    /// overall global manyvector length
    pub global_length: sunindextype,
    /// the C `N_Vector*` array (handles == C pointers)
    pub subvec_array: Vec<N_Vector>,
    /// flag indicating data ownership
    pub own_data: sunbooleantype,
}

pub type N_VectorContent_ManyVector = _N_VectorContent_ManyVector;

/* -----------------------------------------------------------------
   ManyVector content accessor macros
   -----------------------------------------------------------------*/

/// C macro `MANYVECTOR_CONTENT(v)` (mutable borrow of the ManyVector content).
fn content_mut(v: &N_Vector) -> RefMut<'_, N_VectorContent_ManyVector> {
    RefMut::map(v.content.borrow_mut(), |c| {
        c.downcast_mut::<N_VectorContent_ManyVector>()
            .expect("ManyVector N_Vector content")
    })
}

/// C macro `MANYVECTOR_NUM_SUBVECS(v)`.
pub fn MANYVECTOR_NUM_SUBVECS(v: &N_Vector) -> sunindextype {
    content_mut(v).num_subvectors
}

/// C macro `MANYVECTOR_GLOBLENGTH(v)`.
pub fn MANYVECTOR_GLOBLENGTH(v: &N_Vector) -> sunindextype {
    content_mut(v).global_length
}

/// C macro `MANYVECTOR_SUBVECS(v)` — the subvector array as a `RefMut` guard.
/// Drop the guard before calling any op on `v` or on one of its subvectors.
pub fn MANYVECTOR_SUBVECS(v: &N_Vector) -> RefMut<'_, Vec<N_Vector>> {
    RefMut::map(v.content.borrow_mut(), |c| {
        &mut c
            .downcast_mut::<N_VectorContent_ManyVector>()
            .expect("ManyVector N_Vector content")
            .subvec_array
    })
}

/// C macro `MANYVECTOR_SUBVEC(v, i)`. The returned handle is an `Rc` clone —
/// the C pointer copy — and no borrow of `v` outlives this call.
pub fn MANYVECTOR_SUBVEC(v: &N_Vector, i: sunindextype) -> N_Vector {
    content_mut(v).subvec_array[i as usize].clone()
}

/// C macro `MANYVECTOR_OWN_DATA(v)`.
pub fn MANYVECTOR_OWN_DATA(v: &N_Vector) -> sunbooleantype {
    content_mut(v).own_data
}

/* -----------------------------------------------------------------
   ManyVector API routines
   -----------------------------------------------------------------*/

/* This function creates a ManyVector from a set of existing
   N_Vector objects.  ManyVector objects created with this constructor
   may be used to describe data partitioning within a single node. */
pub fn N_VNew_ManyVector(
    num_subvectors: sunindextype,
    vec_array: &[N_Vector],
    sunctx: &SUNContext,
) -> Option<N_Vector> {
    /* Create vector */
    let v = N_VNewEmpty(sunctx)?;

    /* Attach operations */
    {
        let mut ops = v.ops.borrow_mut();

        /* constructors, destructors, and utility operations */
        ops.nvgetvectorid = Some(N_VGetVectorID_ManyVector);
        ops.nvcloneempty = Some(N_VCloneEmpty_ManyVector);
        ops.nvclone = Some(N_VClone_ManyVector);
        ops.nvdestroy = Some(N_VDestroy_ManyVector);
        ops.nvspace = Some(N_VSpace_ManyVector);
        ops.nvgetlength = Some(N_VGetLength_ManyVector);

        /* standard vector operations */
        ops.nvlinearsum = Some(N_VLinearSum_ManyVector);
        ops.nvconst = Some(N_VConst_ManyVector);
        ops.nvprod = Some(N_VProd_ManyVector);
        ops.nvdiv = Some(N_VDiv_ManyVector);
        ops.nvscale = Some(N_VScale_ManyVector);
        ops.nvabs = Some(N_VAbs_ManyVector);
        ops.nvinv = Some(N_VInv_ManyVector);
        ops.nvaddconst = Some(N_VAddConst_ManyVector);
        ops.nvdotprod = Some(N_VDotProdLocal_ManyVector);
        ops.nvmaxnorm = Some(N_VMaxNormLocal_ManyVector);
        ops.nvwrmsnorm = Some(N_VWrmsNorm_ManyVector);
        ops.nvwrmsnormmask = Some(N_VWrmsNormMask_ManyVector);
        ops.nvmin = Some(N_VMinLocal_ManyVector);
        ops.nvwl2norm = Some(N_VWL2Norm_ManyVector);
        ops.nvl1norm = Some(N_VL1NormLocal_ManyVector);
        ops.nvcompare = Some(N_VCompare_ManyVector);
        ops.nvinvtest = Some(N_VInvTestLocal_ManyVector);
        ops.nvconstrmask = Some(N_VConstrMaskLocal_ManyVector);
        ops.nvminquotient = Some(N_VMinQuotientLocal_ManyVector);

        /* fused vector operations */
        ops.nvlinearcombination = Some(N_VLinearCombination_ManyVector);
        ops.nvscaleaddmulti = Some(N_VScaleAddMulti_ManyVector);
        ops.nvdotprodmulti = Some(N_VDotProdMulti_ManyVector);

        /* vector array operations */
        ops.nvwrmsnormvectorarray = Some(N_VWrmsNormVectorArray_ManyVector);
        ops.nvwrmsnormmaskvectorarray = Some(N_VWrmsNormMaskVectorArray_ManyVector);

        /* local reduction operations */
        ops.nvdotprodlocal = Some(N_VDotProdLocal_ManyVector);
        ops.nvmaxnormlocal = Some(N_VMaxNormLocal_ManyVector);
        ops.nvminlocal = Some(N_VMinLocal_ManyVector);
        ops.nvl1normlocal = Some(N_VL1NormLocal_ManyVector);
        ops.nvinvtestlocal = Some(N_VInvTestLocal_ManyVector);
        ops.nvconstrmasklocal = Some(N_VConstrMaskLocal_ManyVector);
        ops.nvminquotientlocal = Some(N_VMinQuotientLocal_ManyVector);
        ops.nvwsqrsumlocal = Some(N_VWSqrSumLocal_ManyVector);
        ops.nvwsqrsummasklocal = Some(N_VWSqrSumMaskLocal_ManyVector);

        /* single buffer reduction operations */
        ops.nvdotprodmultilocal = Some(N_VDotProdMultiLocal_ManyVector);

        /* XBraid interface operations */
        ops.nvbufsize = Some(N_VBufSize_ManyVector);
        ops.nvbufpack = Some(N_VBufPack_ManyVector);
        ops.nvbufunpack = Some(N_VBufUnpack_ManyVector);

        /* debugging functions */
        ops.nvprint = Some(N_VPrint_ManyVector);
        ops.nvprintfile = Some(N_VPrintFile_ManyVector);
    }

    /* Create content */

    /* Attach content components */

    /* allocate and set subvector array */
    let mut subvec_array: Vec<N_Vector> = Vec::with_capacity(num_subvectors as usize);
    for i in 0..num_subvectors {
        subvec_array.push(vec_array[i as usize].clone());
    }

    /* Attach content */
    *v.content.borrow_mut() = Box::new(N_VectorContent_ManyVector {
        num_subvectors,
        global_length: 0,
        subvec_array,
        own_data: SUNFALSE,
    });

    /* Determine overall ManyVector length: sum contributions from all subvectors */
    let mut local_length: sunindextype = 0;
    for i in 0..num_subvectors {
        local_length += N_VGetLength(&vec_array[i as usize]);
    }
    content_mut(&v).global_length = local_length;

    Some(v)
}

/* This function returns the vec_num sub-N_Vector from the N_Vector
   array.  If vec_num is outside of applicable bounds, NULL is returned.
   (Release C compiles the bounds `SUNAssertNull`s out and reads out of
   bounds; deviation class 5 maps that UB to a deterministic panic.) */
pub fn N_VGetSubvector_ManyVector(v: &N_Vector, vec_num: sunindextype) -> N_Vector {
    MANYVECTOR_SUBVEC(v, vec_num)
}

/* This function returns data pointer for the vec_num sub-N_Vector from
   the N_Vector array.  If vec_num is outside of applicable bounds, or if
   the subvector does not support the N_VGetArrayPointer routine, then
   NULL is returned.

   Handle-model mapping: the C `sunrealtype*` is this port's data guard
   (`RefMut<'_, Vec<sunrealtype>>`, see `N_VGetArrayPointer`), and a guard
   borrows the SUBVECTOR handle. The ManyVector keeps its handles inside its
   own `RefCell` content, so the guard cannot be tied to `v`'s lifetime in
   safe Rust; it is therefore handed to `with_data` for the duration of the
   access (`None` exactly where C returns NULL). Writes through the guard
   reach the subvector exactly as writes through the C pointer do. */
pub fn N_VGetSubvectorArrayPointer_ManyVector<R>(
    v: &N_Vector,
    vec_num: sunindextype,
    with_data: impl FnOnce(Option<RefMut<'_, Vec<sunrealtype>>>) -> R,
) -> R {
    let subvec = MANYVECTOR_SUBVEC(v, vec_num);
    let has_getarraypointer = subvec.ops.borrow().nvgetarraypointer.is_some();
    let arr = if has_getarraypointer {
        N_VGetArrayPointer(&subvec)
    } else {
        None
    };
    with_data(arr)
}

/* This function sets the data pointer for the vec_num sub-N_Vector from
   the N_Vector array.  If vec_num is outside of applicable bounds, or if
   the subvector does not support the N_VSetArrayPointer routine, then
   -1 is returned; otherwise this routine returns 0. */
pub fn N_VSetSubvectorArrayPointer_ManyVector(
    v_data: Vec<sunrealtype>,
    v: &N_Vector,
    vec_num: sunindextype,
) -> SUNErrCode {
    N_VSetArrayPointer(v_data, &MANYVECTOR_SUBVEC(v, vec_num));
    SUN_SUCCESS
}

/* This function returns the overall number of sub-vectors.
   It returns a locally stored integer, and is therefore a local call. */
pub fn N_VGetNumSubvectors_ManyVector(v: &N_Vector) -> sunindextype {
    MANYVECTOR_NUM_SUBVECS(v)
}

/* -----------------------------------------------------------------
   ManyVector implementations of generic NVector routines
   -----------------------------------------------------------------*/

/* Returns vector type ID. Used to identify vector implementation
   from abstract N_Vector interface. */
pub fn N_VGetVectorID_ManyVector(_v: &N_Vector) -> N_Vector_ID {
    SUNDIALS_NVEC_MANYVECTOR
}

/* Prints the vector to stdout, calling Print on subvectors. */
pub fn N_VPrint_ManyVector(x: &N_Vector) {
    for i in 0..MANYVECTOR_NUM_SUBVECS(x) {
        N_VPrint(&MANYVECTOR_SUBVEC(x, i));
    }
}

/* Prints the vector to outfile, calling PrintFile on subvectors. */
pub fn N_VPrintFile_ManyVector(x: &N_Vector, outfile: &SUNFile) {
    for i in 0..MANYVECTOR_NUM_SUBVECS(x) {
        N_VPrintFile(&MANYVECTOR_SUBVEC(x, i), outfile);
    }
}

/* Clones a ManyVector, calling CloneEmpty on subvectors. */
pub fn N_VCloneEmpty_ManyVector(w: &N_Vector) -> Option<N_Vector> {
    ManyVectorClone(w, SUNTRUE)
}

/* Clones a ManyVector, calling Clone on subvectors. */
pub fn N_VClone_ManyVector(w: &N_Vector) -> Option<N_Vector> {
    ManyVectorClone(w, SUNFALSE)
}

/* Destroys a ManyVector */
pub fn N_VDestroy_ManyVector(v: N_Vector) {
    /* free content */
    {
        /* C `free(MANYVECTOR_SUBVECS(v)); MANYVECTOR_SUBVECS(v) = NULL;` is
        unconditional — it releases the array of handles either way; the
        pointees are destroyed only when v owns them. */
        let own_data = MANYVECTOR_OWN_DATA(&v);
        let subvec_array: Vec<N_Vector> = std::mem::take(&mut *MANYVECTOR_SUBVECS(&v));

        /* if subvectors are owned by v, then Destroy those */
        if own_data == SUNTRUE {
            for subvec in subvec_array {
                N_VDestroy(subvec);
            }
        }
    }

    /* free ops and vector */
    drop(v);
}

/* Returns the space requirements for the ManyVector, by accumulating this
   information from all subvectors. */
pub fn N_VSpace_ManyVector(v: &N_Vector, lrw: &mut sunindextype, liw: &mut sunindextype) {
    let mut lrw1: sunindextype = 0;
    let mut liw1: sunindextype = 0;
    *lrw = 0;
    *liw = 0;
    for i in 0..MANYVECTOR_NUM_SUBVECS(v) {
        /* update space requirements for this subvector (if 'nvspace' is implemented) */
        let subvec = MANYVECTOR_SUBVEC(v, i);
        let has_nvspace = subvec.ops.borrow().nvspace.is_some();
        if has_nvspace {
            N_VSpace(&subvec, &mut lrw1, &mut liw1);
            *lrw += lrw1;
            *liw += liw1;
        }
    }
}

/* This function retrieves the global length of a ManyVector object. */
pub fn N_VGetLength_ManyVector(v: &N_Vector) -> sunindextype {
    MANYVECTOR_GLOBLENGTH(v)
}

pub fn N_VGetSubvectorLocalLength_ManyVector(
    v: &N_Vector,
    vec_num: sunindextype,
) -> sunindextype {
    let subvector = N_VGetSubvector_ManyVector(v, vec_num);
    N_VGetLocalLength(&subvector)
}

/* Performs the linear sum z = a*x + b*y by calling N_VLinearSum on all subvectors;
   this routine does not check that x, y and z are ManyVectors, if they have the
   same number of subvectors, or if these subvectors are compatible. */
pub fn N_VLinearSum_ManyVector(
    a: sunrealtype,
    x: &N_Vector,
    b: sunrealtype,
    y: &N_Vector,
    z: &N_Vector,
) {
    for i in 0..MANYVECTOR_NUM_SUBVECS(x) {
        N_VLinearSum(
            a,
            &MANYVECTOR_SUBVEC(x, i),
            b,
            &MANYVECTOR_SUBVEC(y, i),
            &MANYVECTOR_SUBVEC(z, i),
        );
    }
}

/* Performs the operation z = c by calling N_VConst on all subvectors. */
pub fn N_VConst_ManyVector(c: sunrealtype, z: &N_Vector) {
    for i in 0..MANYVECTOR_NUM_SUBVECS(z) {
        N_VConst(c, &MANYVECTOR_SUBVEC(z, i));
    }
}

/* Performs the operation z_j = x_j*y_j by calling N_VProd on all subvectors;
   this routine does not check that x, y and z are ManyVectors, if they have the
   same number of subvectors, or if these subvectors are compatible. */
pub fn N_VProd_ManyVector(x: &N_Vector, y: &N_Vector, z: &N_Vector) {
    for i in 0..MANYVECTOR_NUM_SUBVECS(x) {
        N_VProd(
            &MANYVECTOR_SUBVEC(x, i),
            &MANYVECTOR_SUBVEC(y, i),
            &MANYVECTOR_SUBVEC(z, i),
        );
    }
}

/* Performs the operation z_j = x_j/y_j by calling N_VDiv on all subvectors;
   this routine does not check that x, y and z are ManyVectors, if they have the
   same number of subvectors, or if these subvectors are compatible. */
pub fn N_VDiv_ManyVector(x: &N_Vector, y: &N_Vector, z: &N_Vector) {
    for i in 0..MANYVECTOR_NUM_SUBVECS(x) {
        N_VDiv(
            &MANYVECTOR_SUBVEC(x, i),
            &MANYVECTOR_SUBVEC(y, i),
            &MANYVECTOR_SUBVEC(z, i),
        );
    }
}

/* Performs the operation z_j = c*x_j by calling N_VScale on all subvectors;
   this routine does not check that x and z are ManyVectors, if they have the
   same number of subvectors, or if these subvectors are compatible. */
pub fn N_VScale_ManyVector(c: sunrealtype, x: &N_Vector, z: &N_Vector) {
    for i in 0..MANYVECTOR_NUM_SUBVECS(x) {
        N_VScale(c, &MANYVECTOR_SUBVEC(x, i), &MANYVECTOR_SUBVEC(z, i));
    }
}

/* Performs the operation z_j = |x_j| by calling N_VAbs on all subvectors;
   this routine does not check that x and z are ManyVectors, if they have the
   same number of subvectors, or if these subvectors are compatible. */
pub fn N_VAbs_ManyVector(x: &N_Vector, z: &N_Vector) {
    for i in 0..MANYVECTOR_NUM_SUBVECS(x) {
        N_VAbs(&MANYVECTOR_SUBVEC(x, i), &MANYVECTOR_SUBVEC(z, i));
    }
}

/* Performs the operation z_j = 1/x_j by calling N_VInv on all subvectors;
   this routine does not check that x and z are ManyVectors, if they have the
   same number of subvectors, or if these subvectors are compatible. */
pub fn N_VInv_ManyVector(x: &N_Vector, z: &N_Vector) {
    for i in 0..MANYVECTOR_NUM_SUBVECS(x) {
        N_VInv(&MANYVECTOR_SUBVEC(x, i), &MANYVECTOR_SUBVEC(z, i));
    }
}

/* Performs the operation z_j = x_j + b by calling N_VAddConst on all subvectors;
   this routine does not check that x and z are ManyVectors, if they have the
   same number of subvectors, or if these subvectors are compatible. */
pub fn N_VAddConst_ManyVector(x: &N_Vector, b: sunrealtype, z: &N_Vector) {
    for i in 0..MANYVECTOR_NUM_SUBVECS(x) {
        N_VAddConst(&MANYVECTOR_SUBVEC(x, i), b, &MANYVECTOR_SUBVEC(z, i));
    }
}

/* Performs the task-local dot product of two ManyVectors by accumulating
   N_VDotProd over all subvectors; this routine does not check that x and
   y are ManyVectors, if they have the same number of subvectors, or if these
   subvectors are compatible. */
pub fn N_VDotProdLocal_ManyVector(x: &N_Vector, y: &N_Vector) -> sunrealtype {
    let mut sum: sunrealtype;

    /* initialize output*/
    sum = ZERO;

    for i in 0..MANYVECTOR_NUM_SUBVECS(x) {
        /* add subvector contribution */
        sum += N_VDotProd(&MANYVECTOR_SUBVEC(x, i), &MANYVECTOR_SUBVEC(y, i));
    }

    sum
}

/* Performs the task-local maximum norm of a ManyVector by calling
   N_VMaxNormLocal on all subvectors.

   If any subvector does not implement the N_VMaxNormLocal routine (NULL
   function pointer), then this routine will call N_VMaxNorm instead. */
pub fn N_VMaxNormLocal_ManyVector(x: &N_Vector) -> sunrealtype {
    let mut max: sunrealtype;

    /* initialize output*/
    max = ZERO;

    for i in 0..MANYVECTOR_NUM_SUBVECS(x) {
        /* check for nvmaxnormlocal in subvector */
        let subvec = MANYVECTOR_SUBVEC(x, i);
        let has_maxnormlocal = subvec.ops.borrow().nvmaxnormlocal.is_some();
        if has_maxnormlocal {
            let lmax = N_VMaxNormLocal(&subvec);
            max = if max > lmax { max } else { lmax };

        /* otherwise, call nvmaxnorm and accumulate to overall max */
        } else {
            let lmax = N_VMaxNorm(&subvec);
            max = if max > lmax { max } else { lmax };
        }
    }

    max
}

/* Performs the task-local weighted squared sum of a ManyVector by
   unravelling N_VWrmsNorm and N_VGetLength on all subvectors; this routine
   does not check that x and w are ManyVectors, if they have the same number
   of subvectors, or if these subvectors are compatible. */
pub fn N_VWSqrSumLocal_ManyVector(x: &N_Vector, w: &N_Vector) -> sunrealtype {
    let mut n: sunindextype;
    let mut sum: sunrealtype;
    let mut contrib: sunrealtype;

    /* initialize output*/
    sum = ZERO;

    for i in 0..MANYVECTOR_NUM_SUBVECS(x) {
        /* accumulate subvector contribution to overall sum */
        contrib = N_VWrmsNorm(&MANYVECTOR_SUBVEC(x, i), &MANYVECTOR_SUBVEC(w, i));
        n = N_VGetLength(&MANYVECTOR_SUBVEC(x, i));
        sum += contrib * contrib * n as sunrealtype;
    }

    sum
}

/* Performs the WRMS norm of a ManyVector by calling N_VWSqrSumLocal and
   combining the results; this routine does not check that x and
   w are ManyVectors, if they have the same number of subvectors, or if these
   subvectors are compatible. */
pub fn N_VWrmsNorm_ManyVector(x: &N_Vector, w: &N_Vector) -> sunrealtype {
    let gsum: sunrealtype = N_VWSqrSumLocal_ManyVector(x, w);
    SUNRsqrt(gsum / MANYVECTOR_GLOBLENGTH(x) as sunrealtype)
}

/* Performs the task-local masked weighted squared sum of a ManyVector by
   unravelling N_VWrmsNormMask and N_VGetLength on all subvectors; this
   routine does not check that x, w and id are ManyVectors, if they have the
   same number of subvectors, or if these subvectors are compatible. */
pub fn N_VWSqrSumMaskLocal_ManyVector(
    x: &N_Vector,
    w: &N_Vector,
    id: &N_Vector,
) -> sunrealtype {
    let mut n: sunindextype;
    let mut sum: sunrealtype;
    let mut contrib: sunrealtype;

    /* initialize output*/
    sum = ZERO;

    for i in 0..MANYVECTOR_NUM_SUBVECS(x) {
        /* accumulate subvector contribution to overall sum */
        contrib = N_VWrmsNormMask(
            &MANYVECTOR_SUBVEC(x, i),
            &MANYVECTOR_SUBVEC(w, i),
            &MANYVECTOR_SUBVEC(id, i),
        );
        n = N_VGetLength(&MANYVECTOR_SUBVEC(x, i));
        sum += contrib * contrib * n as sunrealtype;
    }

    sum
}

/* Performs the masked WRMS norm of a ManyVector by calling N_VWSqrSumMaskLocal
   and combining the results; this routine does not check that x, w and id are
   ManyVectors, if they have the same number of subvectors, or if these subvectors
   are compatible. */
pub fn N_VWrmsNormMask_ManyVector(x: &N_Vector, w: &N_Vector, id: &N_Vector) -> sunrealtype {
    let gsum: sunrealtype = N_VWSqrSumMaskLocal_ManyVector(x, w, id);
    SUNRsqrt(gsum / MANYVECTOR_GLOBLENGTH(x) as sunrealtype)
}

/* Computes the task-local minimum entry of a ManyVector by calling
   N_VMinLocal on all subvectors.

   If any subvector does not implement the N_VMinLocal routine (NULL
   function pointer), then this routine will call N_VMin instead. */
pub fn N_VMinLocal_ManyVector(x: &N_Vector) -> sunrealtype {
    let mut min: sunrealtype;

    /* initialize output*/
    min = SUN_BIG_REAL;

    for i in 0..MANYVECTOR_NUM_SUBVECS(x) {
        /* check for nvminlocal in subvector */
        let subvec = MANYVECTOR_SUBVEC(x, i);
        let has_minlocal = subvec.ops.borrow().nvminlocal.is_some();
        if has_minlocal {
            let lmin = N_VMinLocal(&subvec);
            min = if min < lmin { min } else { lmin };

        /* otherwise, call nvmin and accumulate to overall min */
        } else {
            let lmin = N_VMin(&subvec);
            min = if min < lmin { min } else { lmin };
        }
    }

    min
}

/* Performs the WL2 norm of a ManyVector by calling N_VSqrSumLocal and
   'massaging' the result.  This routine does not check that x and w are
   ManyVectors, if they have the same number of subvectors, or if these
   subvectors are compatible. */
pub fn N_VWL2Norm_ManyVector(x: &N_Vector, w: &N_Vector) -> sunrealtype {
    let gsum: sunrealtype = N_VWSqrSumLocal_ManyVector(x, w);
    SUNRsqrt(gsum)
}

/* Performs the task-local L1 norm of a ManyVector by accumulating N_VL1Norm
   over all subvectors. */
pub fn N_VL1NormLocal_ManyVector(x: &N_Vector) -> sunrealtype {
    let mut sum: sunrealtype;

    /* initialize output*/
    sum = ZERO;

    for i in 0..MANYVECTOR_NUM_SUBVECS(x) {
        /* accumulate subvector contribution to overall sum */
        sum += N_VL1Norm(&MANYVECTOR_SUBVEC(x, i));
    }

    sum
}

/* Performs N_VCompare on all subvectors; this routine does not check that x and z are
   ManyVectors, if they have the same number of subvectors, or if these subvectors are
   compatible. */
pub fn N_VCompare_ManyVector(c: sunrealtype, x: &N_Vector, z: &N_Vector) {
    for i in 0..MANYVECTOR_NUM_SUBVECS(x) {
        N_VCompare(c, &MANYVECTOR_SUBVEC(x, i), &MANYVECTOR_SUBVEC(z, i));
    }
}

/* Performs the task-local InvTest for a ManyVector by calling N_VInvTestLocal
   on all subvectors and combining the results appropriately.  This routine does
   not check that x and z are ManyVectors, if they have the same number of
   subvectors, or if these subvectors are compatible.

   If any subvector does not implement the N_VInvTestLocal routine (NULL
   function pointer), then this routine will call N_VInvTest instead. */
pub fn N_VInvTestLocal_ManyVector(x: &N_Vector, z: &N_Vector) -> sunbooleantype {
    let mut val: sunbooleantype;

    /* initialize output*/
    val = SUNTRUE;

    for i in 0..MANYVECTOR_NUM_SUBVECS(x) {
        /* check for nvinvtestlocal in subvector */
        let subvec_x = MANYVECTOR_SUBVEC(x, i);
        let has_invtestlocal = subvec_x.ops.borrow().nvinvtestlocal.is_some();
        if has_invtestlocal {
            let subval = N_VInvTestLocal(&subvec_x, &MANYVECTOR_SUBVEC(z, i));
            val = val && subval;

        /* otherwise, call nvinvtest and accumulate to overall val */
        } else {
            let subval = N_VInvTest(&subvec_x, &MANYVECTOR_SUBVEC(z, i));
            val = val && subval;
        }
    }

    val
}

/* Performs the task-local ConstrMask for a ManyVector by calling N_VConstrMaskLocal
   on all subvectors and combining the results appropriately.  This routine does not
   check that c, x and m are ManyVectors, if they have the same number of subvectors,
   or if these subvectors are compatible.

   If any subvector does not implement the N_VConstrMaskLocal routine (NULL
   function pointer), then this routine will call N_VConstrMask instead. */
pub fn N_VConstrMaskLocal_ManyVector(
    c: &N_Vector,
    x: &N_Vector,
    m: &N_Vector,
) -> sunbooleantype {
    let mut val: sunbooleantype;

    /* initialize output*/
    val = SUNTRUE;

    for i in 0..MANYVECTOR_NUM_SUBVECS(x) {
        /* check for nvconstrmasklocal in subvector */
        let subvec_x = MANYVECTOR_SUBVEC(x, i);
        let has_constrmasklocal = subvec_x.ops.borrow().nvconstrmasklocal.is_some();
        if has_constrmasklocal {
            let subval = N_VConstrMaskLocal(
                &MANYVECTOR_SUBVEC(c, i),
                &subvec_x,
                &MANYVECTOR_SUBVEC(m, i),
            );
            val = val && subval;

        /* otherwise, call nvconstrmask and accumulate to overall val */
        } else {
            let subval = N_VConstrMask(
                &MANYVECTOR_SUBVEC(c, i),
                &subvec_x,
                &MANYVECTOR_SUBVEC(m, i),
            );
            val = val && subval;
        }
    }

    val
}

/* Performs the task-local MinQuotient for a ManyVector by calling N_VMinQuotientLocal
   on all subvectors and combining the results appropriately.  This routine does not check
   that num and denom are ManyVectors, if they have the same number of subvectors, or if
   these subvectors are compatible.

   If any subvector does not implement the N_VMinQuotientLocal routine (NULL
   function pointer), then this routine will call N_VMinQuotient instead. */
pub fn N_VMinQuotientLocal_ManyVector(num: &N_Vector, denom: &N_Vector) -> sunrealtype {
    let mut min: sunrealtype;

    /* initialize output*/
    min = SUN_BIG_REAL;

    for i in 0..MANYVECTOR_NUM_SUBVECS(num) {
        /* check for nvminquotientlocal in subvector */
        let subvec_num = MANYVECTOR_SUBVEC(num, i);
        let has_minquotientlocal = subvec_num.ops.borrow().nvminquotientlocal.is_some();
        if has_minquotientlocal {
            let lmin = N_VMinQuotientLocal(&subvec_num, &MANYVECTOR_SUBVEC(denom, i));
            min = if min < lmin { min } else { lmin };

        /* otherwise, call nvmin and accumulate to overall min */
        } else {
            let lmin = N_VMinQuotient(&subvec_num, &MANYVECTOR_SUBVEC(denom, i));
            min = if min < lmin { min } else { lmin };
        }
    }

    min
}

/* -----------------------------------------------------------------
   Single buffer reduction operations
   ----------------------------------------------------------------- */

pub fn N_VDotProdMultiLocal_ManyVector(
    nvec: i32,
    x: &N_Vector,
    Y: &[N_Vector],
    dotprods: &mut [sunrealtype],
) -> SUNErrCode {
    /* create temporary workspace arrays */
    let mut Ysub: Vec<N_Vector> = Vec::with_capacity(nvec as usize);
    let mut contrib: Vec<sunrealtype> = vec![ZERO; nvec as usize];

    /* initialize output */
    for j in 0..nvec as usize {
        dotprods[j] = ZERO;
    }

    /* loop over subvectors */
    for i in 0..MANYVECTOR_NUM_SUBVECS(x) {
        /* extract subvectors from vector array */
        Ysub.clear();
        for j in 0..nvec as usize {
            Ysub.push(MANYVECTOR_SUBVEC(&Y[j], i));
        }

        /* compute dot products */
        let _ = N_VDotProdMultiLocal(nvec, &MANYVECTOR_SUBVEC(x, i), &Ysub, &mut contrib);

        /* accumulate contributions */
        for j in 0..nvec as usize {
            dotprods[j] += contrib[j];
        }
    }

    /* return with success */
    SUN_SUCCESS
}

/* -----------------------------------------------------------------
   Fused vector operations
   ----------------------------------------------------------------- */

/* Performs the linear combination z = sum_j c[j]*X[j] by calling
   N_VLinearCombination on all subvectors; this routine does not check that z
   or the components of X are ManyVectors, if they have the same number of
   subvectors, or if these subvectors are compatible.

   NOTE: implementation of this routine is more challenging, due to the
   array-of-arrays of N_Vectors that comprise X.  This routine will be
   passed an array of ManyVectors, so to call the subvector-specific routines
   we must unravel the subvectors while retaining an array of outer vectors. */
pub fn N_VLinearCombination_ManyVector(
    nvec: i32,
    c: &[sunrealtype],
    X: &[N_Vector],
    z: &N_Vector,
) -> SUNErrCode {
    /* create array of nvec N_Vector pointers for reuse within loop */
    let mut Xsub: Vec<N_Vector> = Vec::with_capacity(nvec as usize);

    /* perform operation by calling N_VLinearCombination for each subvector */
    for i in 0..MANYVECTOR_NUM_SUBVECS(z) {
        /* for each subvector, create the array of subvectors of X */
        Xsub.clear();
        for j in 0..nvec as usize {
            Xsub.push(MANYVECTOR_SUBVEC(&X[j], i));
        }

        /* now call N_VLinearCombination for this array of subvectors */
        let _ = N_VLinearCombination(nvec, c, &Xsub, &MANYVECTOR_SUBVEC(z, i));
    }

    /* clean up and return */
    SUN_SUCCESS
}

/* Performs the ScaleAddMulti operation by calling N_VScaleAddMulti on all
   subvectors; this routine does not check that x, or the components of X and Z are
   ManyVectors, if they have the same number of subvectors, or if these subvectors
   are compatible.

   NOTE: this routine is more challenging, due to the array-of-arrays of
   N_Vectors that comprise Y and Z.  This routine will be passed an array of
   ManyVectors, so to call the subvector-specific routines we must unravel
   the subvectors while retaining an array of outer vectors. */
pub fn N_VScaleAddMulti_ManyVector(
    nvec: i32,
    a: &[sunrealtype],
    x: &N_Vector,
    Y: &[N_Vector],
    Z: &[N_Vector],
) -> SUNErrCode {
    /* create arrays of nvec N_Vector pointers for reuse within loop */
    let mut Ysub: Vec<N_Vector> = Vec::with_capacity(nvec as usize);
    let mut Zsub: Vec<N_Vector> = Vec::with_capacity(nvec as usize);

    /* perform operation by calling N_VScaleAddMulti for each subvector */
    for i in 0..MANYVECTOR_NUM_SUBVECS(x) {
        /* for each subvector, create the array of subvectors of Y and Z */
        Ysub.clear();
        Zsub.clear();
        for j in 0..nvec as usize {
            Ysub.push(MANYVECTOR_SUBVEC(&Y[j], i));
            Zsub.push(MANYVECTOR_SUBVEC(&Z[j], i));
        }

        /* now call N_VScaleAddMulti for this array of subvectors */
        let _ = N_VScaleAddMulti(nvec, a, &MANYVECTOR_SUBVEC(x, i), &Ysub, &Zsub);
    }

    /* clean up and return */
    SUN_SUCCESS
}

/* Performs the DotProdMulti operation by calling N_VDotProdLocal and combining results.
   This routine does not check that x, or the components of Y, are ManyVectors, if
   they have the same number of subvectors, or if these subvectors are compatible. */
pub fn N_VDotProdMulti_ManyVector(
    nvec: i32,
    x: &N_Vector,
    Y: &[N_Vector],
    dotprods: &mut [sunrealtype],
) -> SUNErrCode {
    /* call N_VDotProdLocal for each <x,Y[i]> pair */
    for i in 0..nvec as usize {
        dotprods[i] = N_VDotProdLocal(x, &Y[i]);
    }

    /* return with success */
    SUN_SUCCESS
}

/* -----------------------------------------------------------------
   Vector array operations
   ----------------------------------------------------------------- */

/* Performs the LinearSumVectorArray operation by calling N_VLinearSumVectorArray
   on all subvectors; this routine does not check that the components of X, Y or Z are
   ManyVectors, if they have the same number of subvectors, or if these subvectors
   are compatible.

   NOTE: this routine is more challenging, due to the array-of-arrays of
   N_Vectors that comprise X, Y and Z.  This routine will be passed arrays of
   ManyVectors, so to call the subvector-specific routines we must unravel
   the subvectors while retaining arrays of outer vectors. */
pub fn N_VLinearSumVectorArray_ManyVector(
    nvec: i32,
    a: sunrealtype,
    X: &[N_Vector],
    b: sunrealtype,
    Y: &[N_Vector],
    Z: &[N_Vector],
) -> SUNErrCode {
    /* create arrays of nvec N_Vector pointers for reuse within loop */
    let mut Xsub: Vec<N_Vector> = Vec::with_capacity(nvec as usize);
    let mut Ysub: Vec<N_Vector> = Vec::with_capacity(nvec as usize);
    let mut Zsub: Vec<N_Vector> = Vec::with_capacity(nvec as usize);

    /* perform operation by calling N_VLinearSumVectorArray for each subvector */
    for i in 0..MANYVECTOR_NUM_SUBVECS(&X[0]) {
        /* for each subvector, create the array of subvectors of X, Y and Z */
        Xsub.clear();
        Ysub.clear();
        Zsub.clear();
        for j in 0..nvec as usize {
            Xsub.push(MANYVECTOR_SUBVEC(&X[j], i));
            Ysub.push(MANYVECTOR_SUBVEC(&Y[j], i));
            Zsub.push(MANYVECTOR_SUBVEC(&Z[j], i));
        }

        /* now call N_VLinearSumVectorArray for this array of subvectors */
        let _ = N_VLinearSumVectorArray(nvec, a, &Xsub, b, &Ysub, &Zsub);
    }

    /* clean up and return */
    SUN_SUCCESS
}

/* Performs the ScaleVectorArray operation by calling N_VScaleVectorArray
   on all subvectors; this routine does not check that the components of X or Z are
   ManyVectors, if they have the same number of subvectors, or if these subvectors
   are compatible.

   NOTE: this routine is more challenging, due to the array-of-arrays of
   N_Vectors that comprise X and Z.  This routine will be passed arrays of
   ManyVectors, so to call the subvector-specific routines we must unravel
   the subvectors while retaining arrays of outer vectors. */
pub fn N_VScaleVectorArray_ManyVector(
    nvec: i32,
    c: &[sunrealtype],
    X: &[N_Vector],
    Z: &[N_Vector],
) -> SUNErrCode {
    /* create arrays of nvec N_Vector pointers for reuse within loop */
    let mut Xsub: Vec<N_Vector> = Vec::with_capacity(nvec as usize);
    let mut Zsub: Vec<N_Vector> = Vec::with_capacity(nvec as usize);

    /* perform operation by calling N_VScaleVectorArray for each subvector */
    for i in 0..MANYVECTOR_NUM_SUBVECS(&X[0]) {
        /* for each subvector, create the array of subvectors of X, Y and Z */
        Xsub.clear();
        Zsub.clear();
        for j in 0..nvec as usize {
            Xsub.push(MANYVECTOR_SUBVEC(&X[j], i));
            Zsub.push(MANYVECTOR_SUBVEC(&Z[j], i));
        }

        /* now call N_VScaleVectorArray for this array of subvectors */
        let _ = N_VScaleVectorArray(nvec, c, &Xsub, &Zsub);
    }

    /* clean up and return */
    SUN_SUCCESS
}

/* Performs the ConstVectorArray operation by calling N_VConstVectorArray
   on all subvectors.

   NOTE: this routine is more challenging, due to the array-of-arrays of
   N_Vectors that comprise Z.  This routine will be passed an array of
   ManyVectors, so to call the subvector-specific routines we must unravel
   the subvectors while retaining an array of outer vectors. */
pub fn N_VConstVectorArray_ManyVector(
    nvec: i32,
    c: sunrealtype,
    Z: &[N_Vector],
) -> SUNErrCode {
    /* create array of N_Vector pointers for reuse within loop */
    let mut Zsub: Vec<N_Vector> = Vec::with_capacity(nvec as usize);

    /* perform operation by calling N_VConstVectorArray for each subvector */
    for i in 0..MANYVECTOR_NUM_SUBVECS(&Z[0]) {
        /* for each subvector, create the array of subvectors of X, Y and Z */
        Zsub.clear();
        for j in 0..nvec as usize {
            Zsub.push(MANYVECTOR_SUBVEC(&Z[j], i));
        }

        /* now call N_VConstVectorArray for this array of subvectors */
        let _ = N_VConstVectorArray(nvec, c, &Zsub);
    }

    /* clean up and return */
    SUN_SUCCESS
}

/* Performs the WrmsNormVectorArray operation by calling N_VWSqrSumLocal and combining
   results.  This routine does not check that the components of X or W are ManyVectors, if
   they have the same number of subvectors, or if these subvectors are compatible. */
pub fn N_VWrmsNormVectorArray_ManyVector(
    nvec: i32,
    X: &[N_Vector],
    W: &[N_Vector],
    nrm: &mut [sunrealtype],
) -> SUNErrCode {
    /* call N_VWSqrSumLocal for each (X[i],W[i]) pair */
    for i in 0..nvec as usize {
        nrm[i] = N_VWSqrSumLocal(&X[i], &W[i]);
    }

    /* accumulate totals */

    /* finish off WRMS norms and return */
    for i in 0..nvec as usize {
        nrm[i] = SUNRsqrt(nrm[i] / MANYVECTOR_GLOBLENGTH(&X[i]) as sunrealtype);
    }

    SUN_SUCCESS
}

/* Performs the WrmsNormMaskVectorArray operation by calling N_VWSqrSumMaskLocal and
   combining results.  This routine does not check that id or the components of X and
   W are ManyVectors, if they have the same number of subvectors, or if these
   subvectors are compatible. */
pub fn N_VWrmsNormMaskVectorArray_ManyVector(
    nvec: i32,
    X: &[N_Vector],
    W: &[N_Vector],
    id: &N_Vector,
    nrm: &mut [sunrealtype],
) -> SUNErrCode {
    /* call N_VWSqrSumMaskLocal for each (X[i],W[i]) pair */
    for i in 0..nvec as usize {
        nrm[i] = N_VWSqrSumMaskLocal(&X[i], &W[i], id);
    }

    /* accumulate totals */

    /* finish off WRMS norms and return */
    for i in 0..nvec as usize {
        nrm[i] = SUNRsqrt(nrm[i] / MANYVECTOR_GLOBLENGTH(&X[i]) as sunrealtype);
    }

    SUN_SUCCESS
}

/* Performs the BufSize operation by calling N_VBufSize for each subvector and
   combining results */
pub fn N_VBufSize_ManyVector(x: &N_Vector, size: &mut sunindextype) -> SUNErrCode {
    /* subvector buffer size */
    let mut subvec_size: sunindextype = 0;

    /* initialize total size */
    *size = 0;

    for i in 0..MANYVECTOR_NUM_SUBVECS(x) {
        /* get buffer sized needed for this subvector */
        let _ = N_VBufSize(&MANYVECTOR_SUBVEC(x, i), &mut subvec_size);

        /* update total buffer size */
        *size += subvec_size;
    }

    SUN_SUCCESS
}

/* Performs the BufPack operation by calling N_VBufPack for each subvector where
   the output buffer is offset by the buffer size used by the the previous
   subvector in the set.

   NOTE (faithful to upstream): the C code recomputes `loc = (char*)buf + offset`
   from the buffer BASE each iteration rather than accumulating, so with three
   or more subvectors the later subvectors overwrite earlier ones. The element
   offset below reproduces that byte arithmetic exactly. */
pub fn N_VBufPack_ManyVector(x: &N_Vector, buf: &mut [sunrealtype]) -> SUNErrCode {
    /* subvector buffer offset */
    let mut offset: sunindextype = 0;

    /* start at the beginning of the output buffer */
    let mut loc: usize = 0;

    for i in 0..MANYVECTOR_NUM_SUBVECS(x) {
        let subvec = MANYVECTOR_SUBVEC(x, i);

        /* pack the output buffer starting at the given buffer location */
        let _ = N_VBufPack(&subvec, &mut buf[loc..]);

        /* get the offset from this subvector */
        let _ = N_VBufSize(&subvec, &mut offset);

        /* update the buffer location for the next vector */
        loc = offset as usize / std::mem::size_of::<sunrealtype>();
    }

    SUN_SUCCESS
}

/* Performs the BufUnpack operation by calling N_VBufUnpack for each subvector
   where the input buffer is offset by the buffer size used by the the previous
   subvector in the set (see the note on `N_VBufPack_ManyVector`). */
pub fn N_VBufUnpack_ManyVector(x: &N_Vector, buf: &[sunrealtype]) -> SUNErrCode {
    /* subvector buffer offset */
    let mut offset: sunindextype = 0;

    /* start at the beginning of the input buffer */
    let mut loc: usize = 0;

    for i in 0..MANYVECTOR_NUM_SUBVECS(x) {
        let subvec = MANYVECTOR_SUBVEC(x, i);

        /* unpack the input buffer starting at the given buffer location */
        let _ = N_VBufUnpack(&subvec, &buf[loc..]);

        /* get the offset from this subvector */
        let _ = N_VBufSize(&subvec, &mut offset);

        /* update the buffer location for the next vector */
        loc = offset as usize / std::mem::size_of::<sunrealtype>();
    }

    SUN_SUCCESS
}

/* -----------------------------------------------------------------
   Enable / Disable fused and vector array operations
   ----------------------------------------------------------------- */

pub fn N_VEnableFusedOps_ManyVector(v: &N_Vector, tf: sunbooleantype) -> SUNErrCode {
    let mut ops = v.ops.borrow_mut();
    if tf {
        /* enable all fused vector operations */
        ops.nvlinearcombination = Some(N_VLinearCombination_ManyVector);
        ops.nvscaleaddmulti = Some(N_VScaleAddMulti_ManyVector);
        ops.nvdotprodmulti = Some(N_VDotProdMulti_ManyVector);
        /* enable all vector array operations */
        ops.nvlinearsumvectorarray = Some(N_VLinearSumVectorArray_ManyVector);
        ops.nvscalevectorarray = Some(N_VScaleVectorArray_ManyVector);
        ops.nvconstvectorarray = Some(N_VConstVectorArray_ManyVector);
        ops.nvwrmsnormvectorarray = Some(N_VWrmsNormVectorArray_ManyVector);
        ops.nvwrmsnormmaskvectorarray = Some(N_VWrmsNormMaskVectorArray_ManyVector);
        ops.nvscaleaddmultivectorarray = None;
        ops.nvlinearcombinationvectorarray = None;
        /* enable single buffer reduction operations */
        ops.nvdotprodmultilocal = Some(N_VDotProdMultiLocal_ManyVector);
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

    /* return success */
    SUN_SUCCESS
}

pub fn N_VEnableLinearCombination_ManyVector(
    v: &N_Vector,
    tf: sunbooleantype,
) -> SUNErrCode {
    /* enable/disable operation */
    v.ops.borrow_mut().nvlinearcombination = if tf {
        Some(N_VLinearCombination_ManyVector)
    } else {
        None
    };

    /* return success */
    SUN_SUCCESS
}

pub fn N_VEnableScaleAddMulti_ManyVector(v: &N_Vector, tf: sunbooleantype) -> SUNErrCode {
    /* enable/disable operation */
    v.ops.borrow_mut().nvscaleaddmulti = if tf {
        Some(N_VScaleAddMulti_ManyVector)
    } else {
        None
    };

    /* return success */
    SUN_SUCCESS
}

pub fn N_VEnableDotProdMulti_ManyVector(v: &N_Vector, tf: sunbooleantype) -> SUNErrCode {
    /* enable/disable operation */
    v.ops.borrow_mut().nvdotprodmulti = if tf {
        Some(N_VDotProdMulti_ManyVector)
    } else {
        None
    };

    /* return success */
    SUN_SUCCESS
}

pub fn N_VEnableLinearSumVectorArray_ManyVector(
    v: &N_Vector,
    tf: sunbooleantype,
) -> SUNErrCode {
    /* enable/disable operation */
    v.ops.borrow_mut().nvlinearsumvectorarray = if tf {
        Some(N_VLinearSumVectorArray_ManyVector)
    } else {
        None
    };

    /* return success */
    SUN_SUCCESS
}

pub fn N_VEnableScaleVectorArray_ManyVector(
    v: &N_Vector,
    tf: sunbooleantype,
) -> SUNErrCode {
    /* enable/disable operation */
    v.ops.borrow_mut().nvscalevectorarray = if tf {
        Some(N_VScaleVectorArray_ManyVector)
    } else {
        None
    };

    /* return success */
    SUN_SUCCESS
}

pub fn N_VEnableConstVectorArray_ManyVector(
    v: &N_Vector,
    tf: sunbooleantype,
) -> SUNErrCode {
    /* enable/disable operation */
    v.ops.borrow_mut().nvconstvectorarray = if tf {
        Some(N_VConstVectorArray_ManyVector)
    } else {
        None
    };

    /* return success */
    SUN_SUCCESS
}

pub fn N_VEnableWrmsNormVectorArray_ManyVector(
    v: &N_Vector,
    tf: sunbooleantype,
) -> SUNErrCode {
    /* enable/disable operation */
    v.ops.borrow_mut().nvwrmsnormvectorarray = if tf {
        Some(N_VWrmsNormVectorArray_ManyVector)
    } else {
        None
    };

    /* return success */
    SUN_SUCCESS
}

pub fn N_VEnableWrmsNormMaskVectorArray_ManyVector(
    v: &N_Vector,
    tf: sunbooleantype,
) -> SUNErrCode {
    /* enable/disable operation */
    v.ops.borrow_mut().nvwrmsnormmaskvectorarray = if tf {
        Some(N_VWrmsNormMaskVectorArray_ManyVector)
    } else {
        None
    };

    /* return success */
    SUN_SUCCESS
}

pub fn N_VEnableDotProdMultiLocal_ManyVector(
    v: &N_Vector,
    tf: sunbooleantype,
) -> SUNErrCode {
    /* enable/disable operation */
    v.ops.borrow_mut().nvdotprodmultilocal = if tf {
        Some(N_VDotProdMultiLocal_ManyVector)
    } else {
        None
    };

    /* return success */
    SUN_SUCCESS
}

/* -----------------------------------------------------------------
   Implementation of utility routines
   -----------------------------------------------------------------*/

/* This function performs a generic clone operation on an input N_Vector.
   Based on the 'cloneempty' flag it will either call "nvclone" or
   "nvcloneempty" when creating subvectors in the cloned vector. */
fn ManyVectorClone(w: &N_Vector, cloneempty: sunbooleantype) -> Option<N_Vector> {
    /* Create vector */
    let v = N_VNewEmpty(&w.sunctx.borrow())?;

    /* Attach operations */
    N_VCopyOps(w, &v);

    /* Create content */

    /* Attach content components */

    /* Set scalar components */
    let num_subvectors = MANYVECTOR_NUM_SUBVECS(w);
    let global_length = MANYVECTOR_GLOBLENGTH(w);

    /* Allocate the subvector array */
    let mut subvec_array: Vec<N_Vector> = Vec::with_capacity(num_subvectors as usize);

    /* Clone vectors into the subvector array */
    for i in 0..num_subvectors {
        if cloneempty {
            subvec_array.push(N_VCloneEmpty(&MANYVECTOR_SUBVEC(w, i))?);
        } else {
            subvec_array.push(N_VClone(&MANYVECTOR_SUBVEC(w, i))?);
        }
    }

    /* Attach content and ops to new vector, and return */
    *v.content.borrow_mut() = Box::new(N_VectorContent_ManyVector {
        num_subvectors,
        global_length,
        subvec_array,
        own_data: SUNTRUE,
    });

    Some(v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nvector_serial::{N_VNew_Serial, NV_DATA_S};
    use crate::sundials_context::{SUNContext, SUNContext_Create};

    fn ctx() -> SUNContext {
        let mut c = None;
        assert_eq!(SUNContext_Create(0, &mut c), SUN_SUCCESS);
        c.expect("context created")
    }

    /// Two serial subvectors of lengths 3 and 2 wrapped in a ManyVector.
    fn build(sunctx: &SUNContext) -> (N_Vector, Vec<N_Vector>) {
        let s0 = N_VNew_Serial(3, sunctx).expect("subvector 0");
        let s1 = N_VNew_Serial(2, sunctx).expect("subvector 1");
        let subvecs = vec![s0, s1];
        let v = N_VNew_ManyVector(2, &subvecs, sunctx).expect("manyvector");
        (v, subvecs)
    }

    #[test]
    fn create_and_length_accounting() {
        let sunctx = ctx();
        let (x, subvecs) = build(&sunctx);

        assert_eq!(N_VGetVectorID(&x), SUNDIALS_NVEC_MANYVECTOR);
        assert_eq!(N_VGetNumSubvectors_ManyVector(&x), 2);
        /* global length is the sum of the subvector lengths */
        assert_eq!(N_VGetLength(&x), 5);
        assert_eq!(N_VGetSubvectorLocalLength_ManyVector(&x, 0), 3);
        assert_eq!(N_VGetSubvectorLocalLength_ManyVector(&x, 1), 2);

        /* N_VNew_ManyVector does not take ownership of the subvectors */
        assert_eq!(MANYVECTOR_OWN_DATA(&x), SUNFALSE);

        /* subvector access hands back the very same handle (C pointer copy) */
        assert!(std::rc::Rc::ptr_eq(
            &N_VGetSubvector_ManyVector(&x, 0),
            &subvecs[0]
        ));
        assert!(std::rc::Rc::ptr_eq(
            &N_VGetSubvector_ManyVector(&x, 1),
            &subvecs[1]
        ));

        /* a clone owns its (freshly cloned) subvectors */
        let y = N_VClone(&x).expect("clone");
        assert_eq!(MANYVECTOR_OWN_DATA(&y), SUNTRUE);
        assert_eq!(N_VGetLength(&y), 5);
        assert!(!std::rc::Rc::ptr_eq(
            &N_VGetSubvector_ManyVector(&y, 0),
            &subvecs[0]
        ));

        /* N_VSpace accumulates over the subvectors: 3 + 2 reals, 1 + 1 ints */
        let mut lrw: sunindextype = 0;
        let mut liw: sunindextype = 0;
        N_VSpace(&x, &mut lrw, &mut liw);
        assert_eq!(lrw, 5);
        assert_eq!(liw, 2);
    }

    #[test]
    fn linear_sum_dotprod_wrmsnorm_min() {
        let sunctx = ctx();
        let (x, subvecs) = build(&sunctx);

        N_VConst(2.0, &x);
        let y = N_VClone(&x).expect("clone");
        N_VConst(3.0, &y);
        let z = N_VClone(&x).expect("clone");

        /* z = x + y, applied subvector-wise */
        N_VLinearSum(1.0, &x, 1.0, &y, &z);
        for i in 0..2 {
            let sub = N_VGetSubvector_ManyVector(&z, i);
            let d = NV_DATA_S(&sub);
            for j in 0..d.len() {
                assert_eq!(d[j], 5.0);
            }
        }

        /* aliased: z = 2*z - x */
        N_VLinearSum(2.0, &z, -1.0, &x, &z);
        {
            let sub = N_VGetSubvector_ManyVector(&z, 1);
            assert_eq!(NV_DATA_S(&sub)[1], 8.0);
        }

        /* dot product sums the subvector contributions in C order: 5 * (2*3) */
        assert_eq!(N_VDotProd(&x, &y), 30.0);
        assert_eq!(N_VMaxNorm(&z), 8.0);
        assert_eq!(N_VL1Norm(&x), 10.0);

        /* wrms of a constant vector v with weight w: |v*w| over the whole
        ManyVector — sqrt((1*1*3 + 1*1*2)/5) == 1 */
        let w = N_VClone(&x).expect("clone");
        N_VConst(0.5, &w);
        assert_eq!(N_VWrmsNorm(&x, &w), 1.0);
        assert_eq!(N_VWSqrSumLocal_ManyVector(&x, &w), 5.0);
        assert_eq!(N_VWL2Norm_ManyVector(&x, &w), SUNRsqrt(5.0));

        /* min reaches into the second subvector */
        NV_DATA_S(&subvecs[1])[0] = -1.5;
        assert_eq!(N_VMin(&x), -1.5);
    }

    #[test]
    fn fused_ops_are_enabled_by_the_constructor() {
        let sunctx = ctx();
        let (x, _subvecs) = build(&sunctx);

        N_VConst(1.0, &x);
        let mut X: Vec<N_Vector> = Vec::new();
        for i in 0..3 {
            let v = N_VClone(&x).expect("clone");
            N_VConst((i + 1) as sunrealtype, &v);
            X.push(v);
        }
        let z = N_VClone(&x).expect("clone");
        let c = [1.0, 2.0, 3.0];

        /* 1*1 + 2*2 + 3*3 = 14 in every entry of every subvector */
        assert_eq!(N_VLinearCombination(3, &c, &X, &z), SUN_SUCCESS);
        for i in 0..2 {
            let sub = N_VGetSubvector_ManyVector(&z, i);
            let d = NV_DATA_S(&sub);
            for j in 0..d.len() {
                assert_eq!(d[j], 14.0);
            }
        }

        /* dot products against the same three vectors: 5 * k */
        let mut dotprods = [0.0; 3];
        assert_eq!(N_VDotProdMulti(3, &x, &X, &mut dotprods), SUN_SUCCESS);
        assert_eq!(dotprods, [5.0, 10.0, 15.0]);
    }
}
