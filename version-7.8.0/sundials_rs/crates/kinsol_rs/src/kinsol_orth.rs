//! Port of `src/kinsol/kinsol_orth.c` (orthogonalization workspace for
//! Anderson acceleration: allocation, `SUNQRData` wiring, and selection
//! of the `SUNQRAdd*` variant — MGS / ICWY / CGS2 / DCGS2, whose loop
//! order and inner-product sequence live in
//! `sundials_core::sundials_iterative`).
//!
//! Mapping notes:
//!
//! * C `malloc`s the (uninitialized) `SUNQRData` record inside the
//!   `if (allocate)` block and fills its fields in the unconditional
//!   branch at the bottom. Rust's `SUNQRData_` has a non-optional
//!   `vtemp: N_Vector`, so the record is CONSTRUCTED in that bottom
//!   branch instead — the same place C first gives every field it
//!   later reads a value. The malloc-failure branch is therefore
//!   unreachable and dropped; the `N_VClone` failure branch (which
//!   really can return `None`) is kept verbatim.
//! * **`temp_array` aliasing.** In C `kin_qr_data->temp_array` is a raw
//!   pointer that ALIASES either `kin_T_aa` (ICWY) or `kin_cv`
//!   (CGS2/DCGS2). Rust's `SUNQRData_::temp_array` is a `Vec` by value,
//!   so:
//!   - **ICWY** — the T matrix must persist across `SUNQRAdd` calls AND
//!     is rewritten in place by `AndersonAccQRDelete` in `kinsol.rs`.
//!     `kin_T_aa` therefore stays the authoritative storage in the mem;
//!     [`kinQRAdd`] moves it into `qr_data.temp_array` for the duration
//!     of the call and moves it straight back. `KINInitOrth` leaves
//!     `temp_array` empty (the C pointer assignment has no Rust
//!     counterpart).
//!   - **CGS2 / DCGS2** — `temp_array` is PURE SCRATCH there
//!     (`SUNQRAdd_CGS2` / `_DCGS2` / `_DCGS2_SB` fill every element they
//!     read via `N_VDotProdMulti*` before reading it), and `AndersonAcc`
//!     likewise rewrites `cv[0..nvec]` from index 0 at each entry. The
//!     two uses never observe each other's bytes, so `qr_data` gets its
//!     OWN buffer of exactly `kin_cv`'s length, `2 * (m_aa + 1)`. This
//!     removes the alias without any observable difference and keeps
//!     `kin_cv` free for `AndersonAcc`'s fused-operation locals.
//!     (Accepted deviation class 2 — ownership snapshot.)
//!   - **MGS** — `SUNQRAdd_MGS` never touches `temp_array`; C leaves the
//!     malloc'ed field uninitialized, the port leaves it empty.
//! * `kin_qr_func` is a plain `fn` pointer; the C `(SUNQRAddFn)` casts
//!   are identity casts here since every `SUNQRAdd*` already has the
//!   `SUNQRAddFn` shape.
//! * `KINFreeOrth` does not clear `kin_qr_func` — neither does C.

use sundials_core::sundials_iterative::{
    SUNQRAdd_CGS2, SUNQRAdd_DCGS2, SUNQRAdd_DCGS2_SB, SUNQRAdd_ICWY, SUNQRAdd_ICWY_SB,
    SUNQRAdd_MGS, SUNQRData_,
};
use sundials_core::sundials_nvector::{N_VClone, N_VDestroy, N_Vector};
use sundials_core::sundials_types::*;

use crate::kinsol_impl::{
    KINMem, KINProcessError, KIN_MEM_FAIL, KIN_ORTH_CGS2, KIN_ORTH_DCGS2, KIN_ORTH_ICWY,
    KIN_ORTH_MGS, KIN_SUCCESS, MSG_MEM_FAIL,
};

pub fn KINInitOrth(kin_mem: &KINMem) -> i32 {
    // Do we need to (re)allocate the orthogonalization workspace?
    let (m_aa, orth_aa, allocate) = {
        let m = kin_mem.borrow();
        let allocate: sunbooleantype = m.kin_m_aa > m.kin_orth_aa_alloc;
        (m.kin_m_aa, m.kin_orth_aa, allocate)
    };

    if allocate {
        // Free any existing workspace allocations
        KINFreeOrth(kin_mem);

        // Template vector for creating clones
        let tmpl: N_Vector = kin_mem
            .borrow()
            .kin_unew
            .clone()
            .expect("KINInitOrth: kin_unew (C: cloning from a NULL template is UB)");

        // Update the AA workspace size
        kin_mem.borrow_mut().kin_orth_aa_alloc = m_aa;

        /* Structure of orthogonalization data for QR solve: constructed
        with its fields below (see the module notes) */

        if orth_aa != KIN_ORTH_MGS {
            match N_VClone(&tmpl) {
                Some(v) => kin_mem.borrow_mut().kin_vtemp3 = Some(v), // Orth owns
                None => {
                    KINFreeOrth(kin_mem);
                    KINProcessError(
                        Some(kin_mem),
                        0,
                        line!() as i32,
                        "KINInitOrth",
                        file!(),
                        MSG_MEM_FAIL,
                    );
                    return KIN_MEM_FAIL;
                }
            }
        }

        if orth_aa == KIN_ORTH_ICWY {
            // T matrix for ICWY
            kin_mem.borrow_mut().kin_T_aa = vec![0.0; (m_aa * m_aa) as usize];
        }
    }

    // Does the vector support dot product with single buffer reductions
    {
        let unew: N_Vector = kin_mem
            .borrow()
            .kin_unew
            .clone()
            .expect("KINInitOrth: kin_unew");

        let has_sb = {
            let ops = unew.ops.borrow();
            (ops.nvdotprodlocal.is_some() || ops.nvdotprodmultilocal.is_some())
                && ops.nvdotprodmultiallreduce.is_some()
        };

        let mut m = kin_mem.borrow_mut();
        m.kin_dot_prod_sb = SUNFALSE;
        if has_sb {
            m.kin_dot_prod_sb = SUNTRUE;
        }
    }

    // Initialize the QRData and set the QRAdd function
    let (vtemp2, vtemp3, dot_prod_sb) = {
        let m = kin_mem.borrow();
        (m.kin_vtemp2.clone(), m.kin_vtemp3.clone(), m.kin_dot_prod_sb)
    };

    if orth_aa == KIN_ORTH_MGS {
        let mut m = kin_mem.borrow_mut();
        m.kin_qr_func = Some(SUNQRAdd_MGS);
        m.kin_qr_data = Some(Box::new(SUNQRData_ {
            vtemp: vtemp2.expect("KINInitOrth: kin_vtemp2"), // KINSOL owns
            vtemp2: None,
            temp_array: Vec::new(),
        }));
    } else if orth_aa == KIN_ORTH_ICWY {
        let mut m = kin_mem.borrow_mut();
        if dot_prod_sb {
            m.kin_qr_func = Some(SUNQRAdd_ICWY_SB);
        } else {
            m.kin_qr_func = Some(SUNQRAdd_ICWY);
        }
        m.kin_qr_data = Some(Box::new(SUNQRData_ {
            vtemp: vtemp2.expect("KINInitOrth: kin_vtemp2"), // KINSOL owns
            vtemp2: vtemp3,                                  // Orth owns
            temp_array: Vec::new(),                          // aliases kin_T_aa; see kinQRAdd
        }));
    } else if orth_aa == KIN_ORTH_CGS2 {
        let mut m = kin_mem.borrow_mut();
        m.kin_qr_func = Some(SUNQRAdd_CGS2);
        m.kin_qr_data = Some(Box::new(SUNQRData_ {
            vtemp: vtemp2.expect("KINInitOrth: kin_vtemp2"), // KINSOL owns
            vtemp2: vtemp3,                                  // Orth owns
            /* C: = kin_cv (AA owns); scratch-only, see module notes */
            temp_array: vec![0.0; (2 * (m_aa + 1)) as usize],
        }));
    } else if orth_aa == KIN_ORTH_DCGS2 {
        let mut m = kin_mem.borrow_mut();
        if dot_prod_sb {
            m.kin_qr_func = Some(SUNQRAdd_DCGS2_SB);
        } else {
            m.kin_qr_func = Some(SUNQRAdd_DCGS2);
        }
        m.kin_qr_data = Some(Box::new(SUNQRData_ {
            vtemp: vtemp2.expect("KINInitOrth: kin_vtemp2"), // KINSOL owns
            vtemp2: vtemp3,                                  // Orth owns
            /* C: = kin_cv (AA owns); scratch-only, see module notes */
            temp_array: vec![0.0; (2 * (m_aa + 1)) as usize],
        }));
    }

    KIN_SUCCESS
}

pub fn KINFreeOrth(kin_mem: &KINMem) {
    {
        let mut m = kin_mem.borrow_mut();
        if m.kin_qr_data.is_some() {
            /* C: free(kin_qr_data) — releases the record only; the
            vectors it points at belong to KINSOL / Orth and are
            destroyed separately (below, or by KINFreeVectors) */
            m.kin_qr_data = None;
        }
    }

    let vtemp3 = kin_mem.borrow_mut().kin_vtemp3.take();
    if let Some(v) = vtemp3 {
        N_VDestroy(v);
    }

    {
        let mut m = kin_mem.borrow_mut();

        if !m.kin_T_aa.is_empty() {
            m.kin_T_aa = Vec::new();
        }

        // Reset AA workspace size
        m.kin_orth_aa_alloc = 0;
    }
}

/// The single upstream `kin_qr_func` call site (`AndersonAcc` in
/// `kinsol.c`):
///
/// ```text
/// kin_mem->kin_qr_func(kin_mem->kin_q_aa, R, df, m, mMax,
///                      (void*)kin_mem->kin_qr_data);
/// ```
///
/// Call THIS rather than reaching for `kin_qr_func` / `kin_qr_data`
/// directly: it owns the `temp_array` aliasing contract described in the
/// module notes (for `KIN_ORTH_ICWY` it moves `kin_T_aa` into the
/// `SUNQRData` record for the duration of the call and moves it back
/// afterwards, so `AndersonAccQRDelete`'s direct `kin_T_aa` updates and
/// the QRAdd routine share one T matrix, as the C pointer alias does).
///
/// The record itself is `Option::take`n out of the mem for the call and
/// restored on every path (the QRAdd routines run vector operations that
/// may re-enter the mem).
pub fn kinQRAdd(
    kin_mem: &KINMem,
    Q: &[N_Vector],
    R: &mut [sunrealtype],
    df: &N_Vector,
    m: i32,
    mMax: i32,
) -> SUNErrCode {
    let (qr_func, qr_data, orth_aa) = {
        let mut mem = kin_mem.borrow_mut();
        let qr_func = mem.kin_qr_func;
        let qr_data = mem.kin_qr_data.take();
        let orth_aa = mem.kin_orth_aa;
        (qr_func, qr_data, orth_aa)
    };

    /* C would call through the function pointer unconditionally */
    let qr_func = qr_func.expect("kinQRAdd: kin_qr_func = NULL");
    let mut qr_data = qr_data.expect("kinQRAdd: kin_qr_data = NULL");

    /* ICWY: qrdata->temp_array aliases kin_T_aa in C */
    if orth_aa == KIN_ORTH_ICWY {
        qr_data.temp_array = std::mem::take(&mut kin_mem.borrow_mut().kin_T_aa);
    }

    let ier = qr_func(Q, R, df, m, mMax, &mut qr_data);

    if orth_aa == KIN_ORTH_ICWY {
        kin_mem.borrow_mut().kin_T_aa = std::mem::take(&mut qr_data.temp_array);
    }
    kin_mem.borrow_mut().kin_qr_data = Some(qr_data);

    ier
}
