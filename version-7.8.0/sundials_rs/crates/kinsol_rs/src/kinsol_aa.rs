//! Port of `src/kinsol/kinsol_aa.c` (Anderson acceleration workspace
//! allocation / deallocation utilities).
//!
//! Mapping notes:
//!
//! * C `malloc`-ed `sunrealtype*` arrays (`kin_gamma_aa`, `kin_R_aa`,
//!   `kin_cv`) become `Vec<sunrealtype>`; an EMPTY `Vec` is the `NULL`
//!   state that `KINFreeAA` restores and that the `if (ptr)` guards
//!   test. `Vec` allocation has no `NULL` return, so the C
//!   `MSG_MEM_FAIL` branch for those three is unreachable and is dropped
//!   (the `N_VClone*` branches — which really can return `None` — are
//!   kept verbatim).
//! * C `malloc` leaves the arrays UNINITIALIZED. The port zero-fills.
//!   Every element the algorithm reads is written first (`gamma_aa` by
//!   the back-substitution loop, `R_aa` only in the upper triangle it
//!   just wrote, `cv` from index 0 at every `AndersonAcc` entry), so the
//!   reference build's results do not depend on the initial bytes —
//!   they could not, or its printed output would not be reproducible.
//! * `kin_Xv` is C's `malloc`-ed `N_Vector*` scratch array of pointers.
//!   `Vec<N_Vector>` cannot hold uninitialized handles, so it is filled
//!   with `tmpl` clones (Rc clones = C pointer copies) as placeholders;
//!   `AndersonAcc` overwrites `Xv[0..nvec]` before every use, exactly as
//!   it must in C where the same slots start as garbage.
//! * `KINFreeAA` passes `kin_m_aa` (not `kin_m_aa_alloc`) as the destroy
//!   count, mirroring C; the Rust `N_VDestroyVectorArray` ignores the
//!   count because the `Vec` carries its own length, so the latent C
//!   mismatch when `m_aa` shrank between allocations cannot bite here.

use sundials_core::sundials_nvector::{
    N_VClone, N_VCloneVectorArray, N_VDestroy, N_VDestroyVectorArray, N_Vector,
};
use sundials_core::sundials_types::*;

use crate::kinsol_impl::{KINMem, KINProcessError, KIN_MEM_FAIL, KIN_SUCCESS, MSG_MEM_FAIL};

pub fn KINInitAA(kin_mem: &KINMem) -> i32 {
    // Limit the acceleration space size
    {
        let mut m = kin_mem.borrow_mut();
        if m.kin_m_aa >= m.kin_mxiter {
            m.kin_m_aa = m.kin_mxiter - 1;
        }

        // Initialize the current depth
        m.kin_current_depth = 0;
    }

    // Do we need to (re)allocate the AA workspace?
    let (m_aa, allocate) = {
        let m = kin_mem.borrow();
        let allocate: sunbooleantype = m.kin_m_aa > m.kin_m_aa_alloc;
        (m.kin_m_aa, allocate)
    };

    if allocate {
        // Free any existing workspace allocations
        KINFreeAA(kin_mem);

        // Template vector for creating clones
        let tmpl: N_Vector = kin_mem
            .borrow()
            .kin_unew
            .clone()
            .expect("KINInitAA: kin_unew (C: cloning from a NULL template is UB)");

        // Update the AA workspace size
        kin_mem.borrow_mut().kin_m_aa_alloc = m_aa;

        // Array of acceleration weights
        kin_mem.borrow_mut().kin_gamma_aa = vec![0.0; m_aa as usize];

        // R matrix for QR factorization
        kin_mem.borrow_mut().kin_R_aa = vec![0.0; (m_aa * m_aa) as usize];

        // Q matrix for QR factorization
        match N_VCloneVectorArray(m_aa as i32, &tmpl) {
            Some(v) => kin_mem.borrow_mut().kin_q_aa = v,
            None => {
                KINFreeAA(kin_mem);
                KINProcessError(
                    Some(kin_mem),
                    0,
                    line!() as i32,
                    "KINInitAA",
                    file!(),
                    MSG_MEM_FAIL,
                );
                return KIN_MEM_FAIL;
            }
        }

        // History of residual vector differences
        match N_VCloneVectorArray(m_aa as i32, &tmpl) {
            Some(v) => kin_mem.borrow_mut().kin_df_aa = v,
            None => {
                KINFreeAA(kin_mem);
                KINProcessError(
                    Some(kin_mem),
                    0,
                    line!() as i32,
                    "KINInitAA",
                    file!(),
                    MSG_MEM_FAIL,
                );
                return KIN_MEM_FAIL;
            }
        }

        // History of fixed point function vector differences
        match N_VCloneVectorArray(m_aa as i32, &tmpl) {
            Some(v) => kin_mem.borrow_mut().kin_dg_aa = v,
            None => {
                KINFreeAA(kin_mem);
                KINProcessError(
                    Some(kin_mem),
                    0,
                    line!() as i32,
                    "KINInitAA",
                    file!(),
                    MSG_MEM_FAIL,
                );
                return KIN_MEM_FAIL;
            }
        }

        // Previous residual vector, F(u_{i-1}) = G(u_{i-1}) - u_{i-1}
        match N_VClone(&tmpl) {
            Some(v) => kin_mem.borrow_mut().kin_fold_aa = Some(v),
            None => {
                KINFreeAA(kin_mem);
                KINProcessError(
                    Some(kin_mem),
                    0,
                    line!() as i32,
                    "KINInitAA",
                    file!(),
                    MSG_MEM_FAIL,
                );
                return KIN_MEM_FAIL;
            }
        }

        // Previous fixed point function vector, G(u_{i-1})
        match N_VClone(&tmpl) {
            Some(v) => kin_mem.borrow_mut().kin_gold_aa = Some(v),
            None => {
                KINFreeAA(kin_mem);
                KINProcessError(
                    Some(kin_mem),
                    0,
                    line!() as i32,
                    "KINInitAA",
                    file!(),
                    MSG_MEM_FAIL,
                );
                return KIN_MEM_FAIL;
            }
        }

        // Workspace array for fused operation constants
        kin_mem.borrow_mut().kin_cv = vec![0.0; (2 * (m_aa + 1)) as usize];

        // Workspace array for fused operation vectors
        kin_mem.borrow_mut().kin_Xv = vec![tmpl.clone(); (2 * (m_aa + 1)) as usize];
    }

    KIN_SUCCESS
}

pub fn KINFreeAA(kin_mem: &KINMem) {
    {
        let mut m = kin_mem.borrow_mut();

        if !m.kin_gamma_aa.is_empty() {
            m.kin_gamma_aa = Vec::new();
        }

        if !m.kin_R_aa.is_empty() {
            m.kin_R_aa = Vec::new();
        }
    }

    /* C destroys the vector arrays with the CURRENT kin_m_aa */
    let m_aa = kin_mem.borrow().kin_m_aa as i32;

    let q_aa = std::mem::take(&mut kin_mem.borrow_mut().kin_q_aa);
    if !q_aa.is_empty() {
        N_VDestroyVectorArray(q_aa, m_aa);
    }

    let df_aa = std::mem::take(&mut kin_mem.borrow_mut().kin_df_aa);
    if !df_aa.is_empty() {
        N_VDestroyVectorArray(df_aa, m_aa);
    }

    let dg_aa = std::mem::take(&mut kin_mem.borrow_mut().kin_dg_aa);
    if !dg_aa.is_empty() {
        N_VDestroyVectorArray(dg_aa, m_aa);
    }

    let fold_aa = kin_mem.borrow_mut().kin_fold_aa.take();
    if let Some(v) = fold_aa {
        N_VDestroy(v);
    }

    let gold_aa = kin_mem.borrow_mut().kin_gold_aa.take();
    if let Some(v) = gold_aa {
        N_VDestroy(v);
    }

    {
        let mut m = kin_mem.borrow_mut();

        if !m.kin_cv.is_empty() {
            m.kin_cv = Vec::new();
        }

        if !m.kin_Xv.is_empty() {
            m.kin_Xv = Vec::new();
        }

        // Reset AA workspace size
        m.kin_m_aa_alloc = 0;
    }
}
