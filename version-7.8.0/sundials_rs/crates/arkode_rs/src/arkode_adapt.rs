//! Port of `src/arkode/arkode_adapt.c` (ARKODE's time step adaptivity
//! utilities). The record `ARKodeHAdaptMemRec` and every constant of
//! `arkode_adapt_impl.h` (`ARK_ADAPT_LRW`, `CFLFAC`, `SAFETY`, `GROWTH`,
//! `HFIXED_LB`, `HFIXED_UB`, `ETAMX1`, `ETAMXF`, `ETAMIN`, `ETACF`,
//! `SMALL_NEF`, `PQ`, `ADJUST`) live in the frozen contract
//! (`arkode_impl.rs`), because `arkode_impl.h` `#include`s that header and
//! `ARKodeMemRec` embeds the record.
//!
//! Binding notes:
//! * C `arkAdapt(ark_mem, hadapt_mem, ycur, tcur, hcur, dsm)` drops its
//!   `hadapt_mem` parameter: the record lives *inside* `ark_mem`
//!   (`ark_mem.hadapt_mem`), so it cannot be passed alongside a borrow of the
//!   mem, and it must STAY there for the duration of the call because
//!   `SUNAdaptController_EstimateStep` on an `ARKUserControl` re-enters
//!   `ark_mem->hadapt_mem->{p,q}`. Every C `hadapt_mem->…` access becomes a
//!   granular scoped borrow of `ark_mem` at exactly the C read/write point,
//!   so the field-read order (and therefore the arithmetic) is unchanged.
//!   This is the same treatment `arkode_splittingstep.rs` gives the C
//!   `step_mem` parameter.
//! * `arkPrintAdaptMem` keeps the C argument list; the C `NULL` test becomes
//!   `Option<&ARKodeHAdaptMemRec>`. Call site:
//!   `arkPrintAdaptMem(ark_mem.borrow().hadapt_mem.as_deref(), outfile)`.
//! * `estab_data` is the callback data token (`void*` -> `Option<Box<dyn
//!   Any>>`): `arkAdapt` `Option::take`s it, calls `expstab`, and restores it
//!   on EVERY path including the error return.
//! * `SUNLogDebug` compiles away at `SUNDIALS_LOGGING_LEVEL=2` and is omitted.
//!
//! Accepted deviation (class 5, unobservable): `arkPrintAdaptMem` cannot
//! reproduce C's `%p` rendering of `estab_data`; it prints a fixed
//! placeholder. No reference example calls `ARKodePrintMem`.

use crate::arkode_impl::*;
use sundials_core::sundials_adaptcontroller::{
    SUNAdaptController_EstimateStep, SUNAdaptController_Write,
};
use sundials_core::sundials_errors::SUN_SUCCESS;
use sundials_core::sundials_math::{SUNRabs, SUNRcopysign, SUNMAX, SUNMIN};
use sundials_core::sundials_nvector::N_Vector;
use sundials_core::sundials_types::*;
use sundials_core::sundials_utils::{sun_format_g, SUNFile};

/*---------------------------------------------------------------
  arkAdaptInit:

  This routine creates and sets default values in an
  ARKodeHAdaptMem structure.  This returns a non-NULL structure
  if no errors occurred, or a NULL value otherwise.
  ---------------------------------------------------------------*/
pub fn arkAdaptInit() -> Option<ARKodeHAdaptMem> {
    /* allocate structure (C `malloc` + NULL check; Box allocation cannot
    fail here, so the NULL return is unreachable) */
    /* initialize values (default parameters are set in ARKodeSetDefaults);
    C `memset(hadapt_mem, 0, sizeof(struct ARKodeHAdaptMemRec))` */
    let mut hadapt_mem: ARKodeHAdaptMem = Box::new(ARKodeHAdaptMemRec {
        etamax: ZERO,
        etamx1: ZERO,
        etamxf: ZERO,
        etamin: ZERO,
        small_nef: 0,
        etacf: ZERO,
        cfl: ZERO,
        safety: ZERO,
        growth: ZERO,
        lbound: ZERO,
        ubound: ZERO,
        p: 0,
        q: 0,
        pq: 0,
        adjust: 0,
        hcontroller: None,
        owncontroller: SUNFALSE,
        expstab: None,
        estab_data: None,
        nst_acc: 0,
        nst_exp: 0,
    });
    hadapt_mem.nst_acc = 0;
    hadapt_mem.nst_exp = 0;
    Some(hadapt_mem)
}

/*---------------------------------------------------------------
  arkPrintAdaptMem

  This routine outputs the time step adaptivity memory structure
  to a specified file pointer.
  ---------------------------------------------------------------*/
pub fn arkPrintAdaptMem(hadapt_mem: Option<&ARKodeHAdaptMemRec>, outfile: &SUNFile) {
    if let Some(hadapt_mem) = hadapt_mem {
        outfile.write_str(&format!(
            "ark_hadapt: etamax = {}\n",
            sun_format_g(hadapt_mem.etamax)
        ));
        outfile.write_str(&format!(
            "ark_hadapt: etamx1 = {}\n",
            sun_format_g(hadapt_mem.etamx1)
        ));
        outfile.write_str(&format!(
            "ark_hadapt: etamxf = {}\n",
            sun_format_g(hadapt_mem.etamxf)
        ));
        outfile.write_str(&format!(
            "ark_hadapt: etamin = {}\n",
            sun_format_g(hadapt_mem.etamin)
        ));
        outfile.write_str(&format!(
            "ark_hadapt: small_nef = {}\n",
            hadapt_mem.small_nef
        ));
        outfile.write_str(&format!(
            "ark_hadapt: etacf = {}\n",
            sun_format_g(hadapt_mem.etacf)
        ));
        outfile.write_str(&format!(
            "ark_hadapt: cfl = {}\n",
            sun_format_g(hadapt_mem.cfl)
        ));
        outfile.write_str(&format!(
            "ark_hadapt: safety = {}\n",
            sun_format_g(hadapt_mem.safety)
        ));
        outfile.write_str(&format!(
            "ark_hadapt: growth = {}\n",
            sun_format_g(hadapt_mem.growth)
        ));
        outfile.write_str(&format!(
            "ark_hadapt: lbound = {}\n",
            sun_format_g(hadapt_mem.lbound)
        ));
        outfile.write_str(&format!(
            "ark_hadapt: ubound = {}\n",
            sun_format_g(hadapt_mem.ubound)
        ));
        outfile.write_str(&format!("ark_hadapt: nst_acc = {}\n", hadapt_mem.nst_acc));
        outfile.write_str(&format!("ark_hadapt: nst_exp = {}\n", hadapt_mem.nst_exp));
        outfile.write_str(&format!("ark_hadapt: pq = {}\n", hadapt_mem.pq));
        outfile.write_str(&format!("ark_hadapt: p = {}\n", hadapt_mem.p));
        outfile.write_str(&format!("ark_hadapt: q = {}\n", hadapt_mem.q));
        outfile.write_str(&format!("ark_hadapt: adjust = {}\n", hadapt_mem.adjust));
        if hadapt_mem.expstab.is_none() {
            outfile.write_str("  ark_hadapt: No explicit stability function supplied\n");
        } else {
            outfile.write_str("  ark_hadapt: User provided explicit stability function\n");
            /* C prints the raw `void* estab_data` with "%p"; a `Box<dyn Any>`
            has no reproducible textual address, so a fixed placeholder is
            printed instead (deviation class 5 -- no reference example calls
            ARKodePrintMem). */
            outfile.write_str(&format!(
                "  ark_hadapt: stability function data pointer = {}\n",
                if hadapt_mem.estab_data.is_some() {
                    "(data)"
                } else {
                    "(nil)"
                }
            ));
        }
        if let Some(hcontroller) = hadapt_mem.hcontroller.as_ref() {
            let _ = SUNAdaptController_Write(hcontroller, outfile);
        }
    }
}

/*---------------------------------------------------------------
  arkAdapt is the time step adaptivity wrapper function.  This
  computes and sets the value of ark_eta inside of the ARKodeMem
  data structure.
  ---------------------------------------------------------------*/
pub fn arkAdapt(
    ark_mem: &ARKodeMem,
    ycur: &N_Vector,
    tcur: sunrealtype,
    hcur: sunrealtype,
    dsm: sunrealtype,
) -> i32 {
    let mut retval: i32;
    let mut h_acc: sunrealtype = ZERO;
    let controller_order: i32;

    /* Return with no stepsize adjustment if the controller is NULL */
    let hcontroller = {
        let m = ark_mem.borrow();
        let hadapt_mem = m.hadapt_mem.as_ref().expect("hadapt_mem");
        hadapt_mem.hcontroller.clone()
    };
    let hcontroller = match hcontroller {
        None => {
            ark_mem.borrow_mut().eta = ONE;
            return ARK_SUCCESS;
        }
        Some(hcontroller) => hcontroller,
    };

    /* Request error-based step size from adaptivity controller */
    {
        let m = ark_mem.borrow();
        let hadapt_mem = m.hadapt_mem.as_ref().expect("hadapt_mem");
        if hadapt_mem.pq == 0 {
            controller_order = hadapt_mem.p + hadapt_mem.adjust;
        } else if hadapt_mem.pq == 1 {
            controller_order = hadapt_mem.q + hadapt_mem.adjust;
        } else {
            controller_order = SUNMIN(hadapt_mem.p, hadapt_mem.q) + hadapt_mem.adjust;
        }
    }
    retval = SUNAdaptController_EstimateStep(
        &hcontroller,
        hcur,
        controller_order,
        dsm,
        &mut h_acc,
    );
    if retval != SUN_SUCCESS {
        arkProcessError(
            Some(ark_mem),
            ARK_CONTROLLER_ERR,
            line!() as i32,
            "arkAdapt",
            file!(),
            "SUNAdaptController_EstimateStep failure.",
        );
        return ARK_CONTROLLER_ERR;
    }

    /* enforce safety factors */
    h_acc *= ark_mem
        .borrow()
        .hadapt_mem
        .as_ref()
        .expect("hadapt_mem")
        .safety;

    /* enforce maximum bound on time step growth */
    {
        let etamax = ark_mem
            .borrow()
            .hadapt_mem
            .as_ref()
            .expect("hadapt_mem")
            .etamax;
        h_acc = SUNMIN(SUNRabs(h_acc), SUNRabs(etamax * hcur));
    }

    /* enforce minimum bound time step reduction */
    {
        let etamin = ark_mem
            .borrow()
            .hadapt_mem
            .as_ref()
            .expect("hadapt_mem")
            .etamin;
        h_acc = SUNMAX(h_acc, SUNRabs(etamin * hcur));
    }

    let expstab = ark_mem
        .borrow()
        .hadapt_mem
        .as_ref()
        .expect("hadapt_mem")
        .expstab;
    if let Some(expstab) = expstab {
        let mut h_cfl = ZERO;
        /* `estab_data` is the C `void*` callback token: take it out of the
        record around the call and restore it on every path */
        let mut estab_data = ark_mem
            .borrow_mut()
            .hadapt_mem
            .as_mut()
            .expect("hadapt_mem")
            .estab_data
            .take();
        retval = expstab(ycur, tcur, &mut h_cfl, &mut estab_data);
        ark_mem
            .borrow_mut()
            .hadapt_mem
            .as_mut()
            .expect("hadapt_mem")
            .estab_data = estab_data;
        if retval != ARK_SUCCESS {
            arkProcessError(
                Some(ark_mem),
                ARK_ILL_INPUT,
                line!() as i32,
                "arkAdapt",
                file!(),
                "Error in explicit stability function.",
            );
            return ARK_ILL_INPUT;
        }

        h_cfl *= ark_mem
            .borrow()
            .hadapt_mem
            .as_ref()
            .expect("hadapt_mem")
            .cfl;

        if h_cfl > ZERO && h_cfl < h_acc {
            ark_mem
                .borrow_mut()
                .hadapt_mem
                .as_mut()
                .expect("hadapt_mem")
                .nst_exp += 1;
            h_acc = h_cfl;
        } else {
            ark_mem
                .borrow_mut()
                .hadapt_mem
                .as_mut()
                .expect("hadapt_mem")
                .nst_acc += 1;
        }
    } else {
        ark_mem
            .borrow_mut()
            .hadapt_mem
            .as_mut()
            .expect("hadapt_mem")
            .nst_acc += 1;
    }

    /* enforce adaptivity bounds to retain Jacobian/preconditioner accuracy */
    if dsm <= ONE {
        let (lbound, ubound) = {
            let m = ark_mem.borrow();
            let hadapt_mem = m.hadapt_mem.as_ref().expect("hadapt_mem");
            (hadapt_mem.lbound, hadapt_mem.ubound)
        };
        if (h_acc > SUNRabs(hcur * lbound * ONEMSM)) && (h_acc < SUNRabs(hcur * ubound * ONEPSM)) {
            h_acc = hcur;
        }
    }
    h_acc = SUNRcopysign(h_acc, hcur);

    {
        let mut m = ark_mem.borrow_mut();

        /* set basic value of ark_eta */
        m.eta = h_acc / hcur;

        /* enforce minimum time step size */
        let hmin = m.hmin;
        m.eta = SUNMAX(m.eta, hmin / SUNRabs(hcur));

        /* enforce maximum time step size */
        let denom = SUNMAX(ONE, SUNRabs(hcur) * m.hmax_inv * m.eta);
        m.eta /= denom;
    }

    retval
}

/*===============================================================
  EOF
  ===============================================================*/
