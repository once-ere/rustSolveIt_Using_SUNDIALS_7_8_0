//! Port of `src/arkode/arkode_root.c` (ARKODE's root-finding-in-time
//! utility). `ARKodeRootMemRec` and the constants of `arkode_root_impl.h`
//! (`ARK_ROOT_LRW`, `ARK_ROOT_LIW`, `HUND`) live in the frozen contract
//! (`arkode_impl.rs`) because `arkode_impl.h` `#include`s that header and
//! `ARKodeMemRec` embeds the record.
//!
//! Binding notes:
//! * `void* arkode_mem` -> `&ARKodeMem`; every `arkode_mem == NULL` guard is
//!   unrepresentable and drops out.
//! * C `sunrealtype*` / `int*` / `sunbooleantype*` heap arrays become `Vec`s
//!   inside the record; C `ptr == NULL` becomes `Vec::is_empty()` (the two
//!   states are used interchangeably in `arkPrintRootMem`, which guards each
//!   loop with the NULL test). `free(p); p = NULL` becomes `p = Vec::new()`.
//! * `root_data` (C: a raw snapshot of `ark_mem->user_data`) stays `None`
//!   per the contract; `ark_call_gfun` hands `gfun` the CURRENT
//!   `ark_mem.user_data` box, taking it out around the call and restoring it
//!   on every path (accepted deviation class 6, identical to CVODE's
//!   `cv_call_gfun`).
//! * `arkRootfind` moves the whole rootfinding array state into locals for
//!   the duration of the Illinois search (the user's `g` runs inside the
//!   loop, so no `RefCell` borrow may be held across it) and writes it back
//!   at one exit point -- the locked pattern, byte-identical to C because the
//!   scalars/arrays C writes on each return path are exactly the ones
//!   restored here.
//! * `malloc` failure branches are unreachable (`Vec` allocation aborts), so
//!   the C `ARK_MEM_FAIL` returns for the six arrays and for `root_mem`
//!   itself are dropped, together with their partial-free cleanup.

use crate::arkode::{arkAllocVec, ARKodeGetDky};
use crate::arkode_impl::*;
use sundials_core::sundials_math::{SUNRabs, SUNRdifferentsign, SUNMAX};
use sundials_core::sundials_nvector::{N_VLinearSum, N_VScale, N_Vector};
use sundials_core::sundials_types::*;
use sundials_core::sundials_utils::{sun_format_g, SUNFile};

/*---------------------------------------------------------------
  Invoke the user's root function with the CURRENT user_data box
  (C passes the `root_data` snapshot; see the module docs).
  ---------------------------------------------------------------*/
fn ark_call_gfun(
    ark_mem: &ARKodeMem,
    t: sunrealtype,
    y: &N_Vector,
    gout: &mut [sunrealtype],
) -> i32 {
    let gfun = ark_mem
        .borrow()
        .root_mem
        .as_ref()
        .expect("root_mem")
        .gfun
        .expect("gfun set");
    let mut user_data = ark_mem.borrow_mut().user_data.take();
    let retval = gfun(t, y, gout, &mut user_data);
    ark_mem.borrow_mut().user_data = user_data;
    retval
}

/*===============================================================
  Exported functions
  ===============================================================*/

/*---------------------------------------------------------------
  ARKodeRootInit:

  ARKodeRootInit initializes a rootfinding problem to be solved
  during the integration of the ODE system.  It loads the root
  function pointer and the number of root functions, notifies
  ARKODE that the "fullrhs" function is required, and allocates
  workspace memory.  The return value is ARK_SUCCESS = 0 if no
  errors occurred, or a negative value otherwise.
  ---------------------------------------------------------------*/
pub fn ARKodeRootInit(arkode_mem: &ARKodeMem, nrtfn: i32, g: Option<ARKRootFn>) -> i32 {
    /* unpack ark_mem: the C `arkode_mem == NULL` branch is unrepresentable */
    let ark_mem = arkode_mem;
    let nrt = if nrtfn < 0 { 0 } else { nrtfn };

    /* Ensure that stepper provides fullrhs function */
    if nrt > 0 {
        if ark_mem.borrow().step_fullrhs.is_none() {
            arkProcessError(
                Some(ark_mem),
                ARK_ILL_INPUT,
                line!() as i32,
                "ARKodeRootInit",
                file!(),
                MSG_ARK_MISSING_FULLRHS,
            );
            return ARK_ILL_INPUT;
        }

        let yn = ark_mem.borrow().yn.clone();
        let mut fn_ = ark_mem.borrow_mut().fn_.take();
        let ok = arkAllocVec(ark_mem, yn.as_ref().expect("yn"), &mut fn_);
        ark_mem.borrow_mut().fn_ = fn_;
        if !ok {
            arkProcessError(
                Some(ark_mem),
                ARK_MEM_FAIL,
                line!() as i32,
                "ARKodeRootInit",
                file!(),
                MSG_ARK_MEM_FAIL,
            );
            return ARK_MEM_FAIL;
        }
    }

    /* If unallocated, allocate rootfinding structure, set defaults, update space */
    if ark_mem.borrow().root_mem.is_none() {
        let mut m = ark_mem.borrow_mut();
        /* C `malloc` (cannot fail here); the fields C does NOT set --
        tlo/thi/trout/ttol/nge -- are indeterminate there and zeroed here.
        None of them is read before ARKODE writes it. */
        m.root_mem = Some(Box::new(ARKodeRootMemRec {
            gfun: None,
            nrtfn: 0,
            iroots: Vec::new(),
            rootdir: Vec::new(),
            tlo: ZERO,
            thi: ZERO,
            trout: ZERO,
            glo: Vec::new(),
            ghi: Vec::new(),
            grout: Vec::new(),
            ttol: ZERO,
            irfnd: 0,
            nge: 0,
            gactive: Vec::new(),
            mxgnull: 1,
            /* C: root_data = ark_mem->user_data (raw snapshot); see docs */
            root_data: None,
        }));

        m.lrw += ARK_ROOT_LRW;
        m.liw += ARK_ROOT_LIW;
    }

    /* If rerunning ARKodeRootInit() with a different number of root
       functions (changing number of gfun components), then free
       currently held memory resources */
    {
        let mut m = ark_mem.borrow_mut();
        let nrtfn_old = m.root_mem.as_ref().expect("root_mem").nrtfn;
        if (nrt != nrtfn_old) && (nrtfn_old > 0) {
            {
                let root_mem = m.root_mem.as_mut().expect("root_mem");
                root_mem.glo = Vec::new();
                root_mem.ghi = Vec::new();
                root_mem.grout = Vec::new();
                root_mem.iroots = Vec::new();
                root_mem.rootdir = Vec::new();
                root_mem.gactive = Vec::new();
            }

            m.lrw -= 3 * (nrtfn_old as i64);
            m.liw -= 3 * (nrtfn_old as i64);
        }
    }

    /* If ARKodeRootInit() was called with nrtfn == 0, then set
       nrtfn to zero and gfun to NULL before returning */
    if nrt == 0 {
        let mut m = ark_mem.borrow_mut();
        let root_mem = m.root_mem.as_mut().expect("root_mem");
        root_mem.nrtfn = nrt;
        root_mem.gfun = None;
        return ARK_SUCCESS;
    }

    /* If rerunning ARKodeRootInit() with the same number of root
       functions (not changing number of gfun components), then
       check if the root function argument has changed */
    /* If g != NULL then return as currently reserved memory
       resources will suffice */
    if nrt == ark_mem.borrow().root_mem.as_ref().expect("root_mem").nrtfn {
        let mut m = ark_mem.borrow_mut();
        let gfun_old = m.root_mem.as_ref().expect("root_mem").gfun;
        /* C compares the root-fn pointers by identity; fn-pointer identity
        in Rust carries the same caveats as C across translation units */
        #[allow(unpredictable_function_pointer_comparisons)]
        if g != gfun_old {
            if g.is_none() {
                {
                    let root_mem = m.root_mem.as_mut().expect("root_mem");
                    root_mem.glo = Vec::new();
                    root_mem.ghi = Vec::new();
                    root_mem.grout = Vec::new();
                    root_mem.iroots = Vec::new();
                    root_mem.rootdir = Vec::new();
                    root_mem.gactive = Vec::new();
                }

                m.lrw -= 3 * (nrt as i64);
                m.liw -= 3 * (nrt as i64);

                drop(m);
                arkProcessError(
                    Some(ark_mem),
                    ARK_ILL_INPUT,
                    line!() as i32,
                    "ARKodeRootInit",
                    file!(),
                    MSG_ARK_NULL_G,
                );
                return ARK_ILL_INPUT;
            } else {
                m.root_mem.as_mut().expect("root_mem").gfun = g;
                return ARK_SUCCESS;
            }
        } else {
            return ARK_SUCCESS;
        }
    }

    /* Set variable values in ARKODE memory block */
    ark_mem
        .borrow_mut()
        .root_mem
        .as_mut()
        .expect("root_mem")
        .nrtfn = nrt;
    if g.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "ARKodeRootInit",
            file!(),
            MSG_ARK_NULL_G,
        );
        return ARK_ILL_INPUT;
    } else {
        ark_mem
            .borrow_mut()
            .root_mem
            .as_mut()
            .expect("root_mem")
            .gfun = g;
    }

    /* Allocate necessary memory and return (the C allocation-failure
    branches are unreachable: Vec allocation aborts rather than returning
    NULL). C leaves glo/ghi/grout/iroots uninitialised. */
    {
        let mut m = ark_mem.borrow_mut();
        {
            let root_mem = m.root_mem.as_mut().expect("root_mem");
            root_mem.glo = vec![ZERO; nrt as usize];
            root_mem.ghi = vec![ZERO; nrt as usize];
            root_mem.grout = vec![ZERO; nrt as usize];
            root_mem.iroots = vec![0; nrt as usize];
            root_mem.rootdir = vec![0; nrt as usize];
            root_mem.gactive = vec![SUNFALSE; nrt as usize];

            /* Set default values for rootdir (both directions) */
            for i in 0..nrt as usize {
                root_mem.rootdir[i] = 0;
            }

            /* Set default values for gactive (all active) */
            for i in 0..nrt as usize {
                root_mem.gactive[i] = SUNTRUE;
            }
        }

        m.lrw += 3 * (nrt as i64);
        m.liw += 3 * (nrt as i64);
    }

    ARK_SUCCESS
}

/*===============================================================
  Private functions
  ===============================================================*/

/*---------------------------------------------------------------
  arkRootFree

  This routine frees all memory associated with ARKODE's
  rootfinding module.
  ---------------------------------------------------------------*/
pub fn arkRootFree(arkode_mem: &ARKodeMem) -> i32 {
    /* the C `arkode_mem == NULL` branch is unrepresentable */
    let ark_mem = arkode_mem;
    let mut m = ark_mem.borrow_mut();
    if m.root_mem.is_some() {
        let nrtfn = m.root_mem.as_ref().expect("root_mem").nrtfn;
        if nrtfn > 0 {
            {
                let root_mem = m.root_mem.as_mut().expect("root_mem");
                root_mem.glo = Vec::new();
                root_mem.ghi = Vec::new();
                root_mem.grout = Vec::new();
                root_mem.iroots = Vec::new();
                root_mem.rootdir = Vec::new();
                root_mem.gactive = Vec::new();
            }
            m.lrw -= 3 * (nrtfn as i64);
            m.liw -= 3 * (nrtfn as i64);
        }
        m.root_mem = None;
        m.lrw -= ARK_ROOT_LRW;
        m.liw -= ARK_ROOT_LIW;
    }
    ARK_SUCCESS
}

/*---------------------------------------------------------------
  arkPrintRootMem

  This routine outputs the root-finding memory structure to a
  specified file pointer.
  ---------------------------------------------------------------*/
pub fn arkPrintRootMem(arkode_mem: &ARKodeMem, outfile: &SUNFile) -> i32 {
    /* the C `arkode_mem == NULL` branch is unrepresentable */
    let ark_mem = arkode_mem;
    let m = ark_mem.borrow();
    if let Some(root_mem) = m.root_mem.as_ref() {
        outfile.write_str(&format!("ark_nrtfn = {}\n", root_mem.nrtfn));
        outfile.write_str(&format!("ark_nge = {}\n", root_mem.nge));
        if !root_mem.iroots.is_empty() {
            for i in 0..root_mem.nrtfn as usize {
                outfile.write_str(&format!("ark_iroots[{}] = {}\n", i, root_mem.iroots[i]));
            }
        }
        if !root_mem.rootdir.is_empty() {
            for i in 0..root_mem.nrtfn as usize {
                outfile.write_str(&format!("ark_rootdir[{}] = {}\n", i, root_mem.rootdir[i]));
            }
        }
        outfile.write_str(&format!("ark_irfnd = {}\n", root_mem.irfnd));
        outfile.write_str(&format!("ark_mxgnull = {}\n", root_mem.mxgnull));
        if !root_mem.gactive.is_empty() {
            for i in 0..root_mem.nrtfn as usize {
                outfile.write_str(&format!(
                    "ark_gactive[{}] = {}\n",
                    i,
                    if root_mem.gactive[i] { 1 } else { 0 }
                ));
            }
        }
        outfile.write_str(&format!("ark_tlo = {}\n", sun_format_g(root_mem.tlo)));
        outfile.write_str(&format!("ark_thi = {}\n", sun_format_g(root_mem.thi)));
        outfile.write_str(&format!("ark_trout = {}\n", sun_format_g(root_mem.trout)));
        if !root_mem.glo.is_empty() {
            for i in 0..root_mem.nrtfn as usize {
                outfile.write_str(&format!(
                    "ark_glo[{}] = {}\n",
                    i,
                    sun_format_g(root_mem.glo[i])
                ));
            }
        }
        if !root_mem.ghi.is_empty() {
            for i in 0..root_mem.nrtfn as usize {
                outfile.write_str(&format!(
                    "ark_ghi[{}] = {}\n",
                    i,
                    sun_format_g(root_mem.ghi[i])
                ));
            }
        }
        if !root_mem.grout.is_empty() {
            for i in 0..root_mem.nrtfn as usize {
                outfile.write_str(&format!(
                    "ark_grout[{}] = {}\n",
                    i,
                    sun_format_g(root_mem.grout[i])
                ));
            }
        }
        outfile.write_str(&format!("ark_ttol = {}\n", sun_format_g(root_mem.ttol)));
    }
    ARK_SUCCESS
}

/*---------------------------------------------------------------
  arkRootCheck1

  This routine completes the initialization of rootfinding memory
  information, and checks whether g has a zero both at and very near
  the initial point of the IVP.

  This routine returns an int equal to:
    ARK_RTFUNC_FAIL < 0  if the g function failed, or
    ARK_SUCCESS     = 0  otherwise.
  ---------------------------------------------------------------*/
pub fn arkRootCheck1(arkode_mem: &ARKodeMem) -> i32 {
    /* the C `arkode_mem == NULL` branch is unrepresentable */
    let ark_mem = arkode_mem;

    {
        let mut m = ark_mem.borrow_mut();
        let tcur = m.tcur;
        let h = m.h;
        let uround = m.uround;
        let root_mem = m.root_mem.as_mut().expect("root_mem");
        for i in 0..root_mem.nrtfn as usize {
            root_mem.iroots[i] = 0;
        }
        root_mem.tlo = tcur;
        root_mem.ttol = (SUNRabs(tcur) + SUNRabs(h)) * uround * HUND;
    }

    /* Evaluate g at initial t and check for zero values. */
    let (tlo, yn) = {
        let m = ark_mem.borrow();
        (
            m.root_mem.as_ref().expect("root_mem").tlo,
            m.yn.clone().expect("yn"),
        )
    };
    let mut glo = std::mem::take(&mut ark_mem.borrow_mut().root_mem.as_mut().expect("root_mem").glo);
    let retval = ark_call_gfun(ark_mem, tlo, &yn, &mut glo);
    {
        let mut m = ark_mem.borrow_mut();
        let root_mem = m.root_mem.as_mut().expect("root_mem");
        root_mem.glo = glo;
        root_mem.nge = 1;
    }
    if retval != 0 {
        let tcur = ark_mem.borrow().tcur;
        arkProcessError(
            Some(ark_mem),
            ARK_RTFUNC_FAIL,
            line!() as i32,
            "arkRootCheck1",
            file!(),
            &MSG_ARK_RTFUNC_FAILED(tcur),
        );
        return ARK_RTFUNC_FAIL;
    }

    let mut zroot = SUNFALSE;
    {
        let mut m = ark_mem.borrow_mut();
        let root_mem = m.root_mem.as_mut().expect("root_mem");
        for i in 0..root_mem.nrtfn as usize {
            if SUNRabs(root_mem.glo[i]) == ZERO {
                zroot = SUNTRUE;
                root_mem.gactive[i] = SUNFALSE;
            }
        }
    }
    if !zroot {
        return ARK_SUCCESS;
    }

    /* call full RHS if needed */
    if !ark_mem.borrow().fn_is_current {
        let (step_fullrhs, tn, yn, fn_) = {
            let m = ark_mem.borrow();
            (
                m.step_fullrhs.expect("step_fullrhs"),
                m.tn,
                m.yn.clone().expect("yn"),
                m.fn_.clone().expect("fn"),
            )
        };
        let retval = step_fullrhs(ark_mem, tn, &yn, &fn_, ARK_FULLRHS_START);
        if retval != 0 {
            let tcur = ark_mem.borrow().tcur;
            arkProcessError(
                Some(ark_mem),
                ARK_RHSFUNC_FAIL,
                line!() as i32,
                "arkRootCheck1",
                file!(),
                &MSG_ARK_RHSFUNC_FAILED(tcur),
            );
            return ARK_RHSFUNC_FAIL;
        }
        ark_mem.borrow_mut().fn_is_current = SUNTRUE;
    }

    /* Some g_i is zero at t0; look at g at t0+(small increment). */
    let (smallh, tplus, yn, fn_, tempv4) = {
        let m = ark_mem.borrow();
        let root_mem = m.root_mem.as_ref().expect("root_mem");
        let hratio = SUNMAX(root_mem.ttol / SUNRabs(m.h), TENTH);
        let smallh = hratio * m.h;
        let tplus = root_mem.tlo + smallh;
        (
            smallh,
            tplus,
            m.yn.clone().expect("yn"),
            m.fn_.clone().expect("fn"),
            m.tempv4.clone().expect("tempv4"),
        )
    };
    N_VLinearSum(ONE, &yn, smallh, &fn_, &tempv4);
    let mut ghi = std::mem::take(&mut ark_mem.borrow_mut().root_mem.as_mut().expect("root_mem").ghi);
    let retval = ark_call_gfun(ark_mem, tplus, &tempv4, &mut ghi);
    {
        let mut m = ark_mem.borrow_mut();
        let root_mem = m.root_mem.as_mut().expect("root_mem");
        root_mem.ghi = ghi;
        root_mem.nge += 1;
    }
    if retval != 0 {
        let tcur = ark_mem.borrow().tcur;
        arkProcessError(
            Some(ark_mem),
            ARK_RTFUNC_FAIL,
            line!() as i32,
            "arkRootCheck1",
            file!(),
            &MSG_ARK_RTFUNC_FAILED(tcur),
        );
        return ARK_RTFUNC_FAIL;
    }

    /* We check now only the components of g which were exactly 0.0 at t0
     * to see if we can 'activate' them. */
    {
        let mut m = ark_mem.borrow_mut();
        let root_mem = m.root_mem.as_mut().expect("root_mem");
        for i in 0..root_mem.nrtfn as usize {
            if !root_mem.gactive[i] && SUNRabs(root_mem.ghi[i]) != ZERO {
                root_mem.gactive[i] = SUNTRUE;
                root_mem.glo[i] = root_mem.ghi[i];
            }
        }
    }
    ARK_SUCCESS
}

/*---------------------------------------------------------------
  arkRootCheck2

  This routine checks for exact zeros of g at the last root found,
  if the last return was a root.  It then checks for a close pair of
  zeros (an error condition), and for a new root at a nearby point.
  The array glo = g(tlo) at the left endpoint of the search interval
  is adjusted if necessary to assure that all g_i are nonzero
  there, before returning to do a root search in the interval.

  On entry, tlo = tretlast is the last value of tret returned by
  ARKODE.  This may be the previous tn, the previous tout value, or
  the last root location.

  This routine returns an int equal to:
    ARK_RTFUNC_FAIL < 0 if the g function failed, or
    CLOSERT         = 3 if a close pair of zeros was found, or
    RTFOUND         = 1 if a new zero of g was found near tlo, or
    ARK_SUCCESS     = 0 otherwise.
  ---------------------------------------------------------------*/
pub fn arkRootCheck2(arkode_mem: &ARKodeMem) -> i32 {
    /* the C `arkode_mem == NULL` branch is unrepresentable */
    let ark_mem = arkode_mem;

    /* return if no roots in previous step */
    if ark_mem.borrow().root_mem.as_ref().expect("root_mem").irfnd == 0 {
        return ARK_SUCCESS;
    }

    /* Set tempv4 = y(tlo) */
    let (tlo, tempv4) = {
        let m = ark_mem.borrow();
        (
            m.root_mem.as_ref().expect("root_mem").tlo,
            m.tempv4.clone().expect("tempv4"),
        )
    };
    let _ = ARKodeGetDky(ark_mem, tlo, 0, &tempv4);

    /* Evaluate root-finding function: glo = g(tlo, y(tlo)) */
    let mut glo = std::mem::take(&mut ark_mem.borrow_mut().root_mem.as_mut().expect("root_mem").glo);
    let retval = ark_call_gfun(ark_mem, tlo, &tempv4, &mut glo);
    {
        let mut m = ark_mem.borrow_mut();
        let root_mem = m.root_mem.as_mut().expect("root_mem");
        root_mem.glo = glo;
        root_mem.nge += 1;
    }
    if retval != 0 {
        return ARK_RTFUNC_FAIL;
    }

    /* reset root-finding flags (overall, and for specific eqns) */
    let mut zroot = SUNFALSE;
    {
        let mut m = ark_mem.borrow_mut();
        let root_mem = m.root_mem.as_mut().expect("root_mem");
        for i in 0..root_mem.nrtfn as usize {
            root_mem.iroots[i] = 0;
        }

        /* for all active roots, check if glo_i == 0 to mark roots found */
        for i in 0..root_mem.nrtfn as usize {
            if !root_mem.gactive[i] {
                continue;
            }
            if SUNRabs(root_mem.glo[i]) == ZERO {
                zroot = SUNTRUE;
                root_mem.iroots[i] = 1;
            }
        }
    }
    if !zroot {
        return ARK_SUCCESS; /* return if no roots */
    }

    /* One or more g_i has a zero at tlo.  Check g at tlo+smallh. */
    /*     set time tolerance */
    /*     set tplus = tlo + smallh */
    let (smallh, tplus, past_tn) = {
        let mut m = ark_mem.borrow_mut();
        let tcur = m.tcur;
        let h = m.h;
        let uround = m.uround;
        let root_mem = m.root_mem.as_mut().expect("root_mem");
        root_mem.ttol = (SUNRabs(tcur) + SUNRabs(h)) * uround * HUND;
        let smallh = if h > ZERO { root_mem.ttol } else { -root_mem.ttol };
        let tplus = root_mem.tlo + smallh;
        (smallh, tplus, (tplus - tcur) * h >= ZERO)
    };
    /*     update ark_ycur with small explicit Euler step (if tplus is past tn) */
    let ycur = ark_mem.borrow().ycur.clone().expect("ycur");
    if past_tn {
        /* hratio = smallh/ark_mem->h; */
        let fn_ = ark_mem.borrow().fn_.clone().expect("fn");
        N_VLinearSum(ONE, &tempv4, smallh, &fn_, &ycur);
    } else {
        /*   set ark_ycur = y(tplus) via interpolation */
        let _ = ARKodeGetDky(ark_mem, tplus, 0, &ycur);
    }
    /*     set ghi = g(tplus,y(tplus)) */
    let mut ghi = std::mem::take(&mut ark_mem.borrow_mut().root_mem.as_mut().expect("root_mem").ghi);
    let retval = ark_call_gfun(ark_mem, tplus, &ycur, &mut ghi);
    {
        let mut m = ark_mem.borrow_mut();
        let root_mem = m.root_mem.as_mut().expect("root_mem");
        root_mem.ghi = ghi;
        root_mem.nge += 1;
    }
    if retval != 0 {
        return ARK_RTFUNC_FAIL;
    }

    /* Check for close roots (error return), for a new zero at tlo+smallh,
    and for a g_i that changed from zero to nonzero. */
    zroot = SUNFALSE;
    {
        let mut m = ark_mem.borrow_mut();
        let root_mem = m.root_mem.as_mut().expect("root_mem");
        for i in 0..root_mem.nrtfn as usize {
            if !root_mem.gactive[i] {
                continue;
            }
            if SUNRabs(root_mem.ghi[i]) == ZERO {
                if root_mem.iroots[i] == 1 {
                    return CLOSERT;
                }
                zroot = SUNTRUE;
                root_mem.iroots[i] = 1;
            } else {
                if root_mem.iroots[i] == 1 {
                    root_mem.glo[i] = root_mem.ghi[i];
                }
            }
        }
    }
    if zroot {
        return RTFOUND;
    }
    ARK_SUCCESS
}

/*---------------------------------------------------------------
  arkRootCheck3

  This routine interfaces to arkRootfind to look for a root of g
  between tlo and either tn or tout, whichever comes first.
  Only roots beyond tlo in the direction of integration are sought.

  This routine returns an int equal to:
    ARK_RTFUNC_FAIL < 0 if the g function failed, or
    RTFOUND         = 1 if a root of g was found, or
    ARK_SUCCESS     = 0 otherwise.
  ---------------------------------------------------------------*/
pub fn arkRootCheck3(arkode_mem: &ARKodeMem, tout: sunrealtype, itask: i32) -> i32 {
    /* the C `arkode_mem == NULL` branch is unrepresentable */
    let ark_mem = arkode_mem;

    let tempv4 = ark_mem.borrow().tempv4.clone().expect("tempv4");

    /* Set thi = tn or tout, whichever comes first; set y = y(thi). */
    if itask == ARK_ONE_STEP {
        let yn = {
            let mut m = ark_mem.borrow_mut();
            let tcur = m.tcur;
            m.root_mem.as_mut().expect("root_mem").thi = tcur;
            m.yn.clone().expect("yn")
        };
        N_VScale(ONE, &yn, &tempv4);
    }
    if itask == ARK_NORMAL {
        let past_tn = {
            let m = ark_mem.borrow();
            (tout - m.tcur) * m.h >= ZERO
        };
        if past_tn {
            let yn = {
                let mut m = ark_mem.borrow_mut();
                let tcur = m.tcur;
                m.root_mem.as_mut().expect("root_mem").thi = tcur;
                m.yn.clone().expect("yn")
            };
            N_VScale(ONE, &yn, &tempv4);
        } else {
            let thi = {
                let mut m = ark_mem.borrow_mut();
                let root_mem = m.root_mem.as_mut().expect("root_mem");
                root_mem.thi = tout;
                root_mem.thi
            };
            let _ = ARKodeGetDky(ark_mem, thi, 0, &tempv4);
        }
    }

    /* Set rootmem->ghi = g(thi) and call arkRootfind to search (tlo,thi) for roots. */
    let thi = ark_mem.borrow().root_mem.as_ref().expect("root_mem").thi;
    let mut ghi = std::mem::take(&mut ark_mem.borrow_mut().root_mem.as_mut().expect("root_mem").ghi);
    let retval = ark_call_gfun(ark_mem, thi, &tempv4, &mut ghi);
    {
        let mut m = ark_mem.borrow_mut();
        let root_mem = m.root_mem.as_mut().expect("root_mem");
        root_mem.ghi = ghi;
        root_mem.nge += 1;
    }
    if retval != 0 {
        return ARK_RTFUNC_FAIL;
    }

    {
        let mut m = ark_mem.borrow_mut();
        let tcur = m.tcur;
        let h = m.h;
        let uround = m.uround;
        m.root_mem.as_mut().expect("root_mem").ttol =
            (SUNRabs(tcur) + SUNRabs(h)) * uround * HUND;
    }
    let ier = arkRootfind(ark_mem);
    if ier == ARK_RTFUNC_FAIL {
        return ARK_RTFUNC_FAIL;
    }
    {
        let mut m = ark_mem.borrow_mut();
        let root_mem = m.root_mem.as_mut().expect("root_mem");
        for i in 0..root_mem.nrtfn as usize {
            if !root_mem.gactive[i] && root_mem.grout[i] != ZERO {
                root_mem.gactive[i] = SUNTRUE;
            }
        }
        root_mem.tlo = root_mem.trout;
        for i in 0..root_mem.nrtfn as usize {
            root_mem.glo[i] = root_mem.grout[i];
        }
    }

    /* If no root found, return ARK_SUCCESS. */
    if ier == ARK_SUCCESS {
        return ARK_SUCCESS;
    }

    /* If a root was found, interpolate to get y(trout) and return.  */
    let (trout, ycur) = {
        let m = ark_mem.borrow();
        (
            m.root_mem.as_ref().expect("root_mem").trout,
            m.ycur.clone().expect("ycur"),
        )
    };
    let _ = ARKodeGetDky(ark_mem, trout, 0, &ycur);
    RTFOUND
}

/*---------------------------------------------------------------
  arkRootfind

  This routine solves for a root of g(t) between tlo and thi, if
  one exists.  Only roots of odd multiplicity (i.e. with a change
  of sign in one of the g_i), or exact zeros, are found.
  Here the sign of tlo - thi is arbitrary, but if multiple roots
  are found, the one closest to tlo is returned.

  The method used is the Illinois algorithm, a modified secant method.
  Reference: Kathie L. Hiebert and Lawrence F. Shampine, Implicitly
  Defined Output Points for Solutions of ODEs, Sandia National
  Laboratory Report SAND80-0180, February 1980.

  This routine uses the following parameters for communication:

  nrtfn    = number of functions g_i, or number of components of
            the vector-valued function g(t).  Input only.

  gfun     = user-defined function for g(t).  Its form is
             (void) gfun(t, y, gt, user_data)

  rootdir  = in array specifying the direction of zero-crossings.
             If rootdir[i] > 0, search for roots of g_i only if
             g_i is increasing; if rootdir[i] < 0, search for
             roots of g_i only if g_i is decreasing; otherwise
             always search for roots of g_i.

  gactive  = array specifying whether a component of g should
             or should not be monitored. gactive[i] is initially
             set to SUNTRUE for all i=0,...,nrtfn-1, but it may be
             reset to SUNFALSE if at the first step g[i] is 0.0
             both at the I.C. and at a small perturbation of them.
             gactive[i] is then set back on SUNTRUE only after the
             corresponding g function moves away from 0.0.

  nge      = cumulative counter for gfun calls.

  ttol     = a convergence tolerance for trout.  Input only.
             When a root at trout is found, it is located only to
             within a tolerance of ttol.  Typically, ttol should
             be set to a value on the order of
               100 * UROUND * max (SUNRabs(tlo), SUNRabs(thi))
             where UROUND is the unit roundoff of the machine.

  tlo, thi = endpoints of the interval in which roots are sought.
             On input, and must be distinct, but tlo - thi may
             be of either sign.  The direction of integration is
             assumed to be from tlo to thi.  On return, tlo and thi
             are the endpoints of the final relevant interval.

  glo, ghi = arrays of length nrtfn containing the vectors g(tlo)
             and g(thi) respectively.  Input and output.  On input,
             none of the glo[i] should be zero.

  trout    = root location, if a root was found, or thi if not.
             Output only.  If a root was found other than an exact
             zero of g, trout is the endpoint thi of the final
             interval bracketing the root, with size at most ttol.

  grout    = array of length nrtfn containing g(trout) on return.

  iroots   = int array of length nrtfn with root information.
             Output only.  If a root was found, iroots indicates
             which components g_i have a root at trout.  For
             i = 0, ..., nrtfn-1, iroots[i] = 1 if g_i has a root
             and g_i is increasing, iroots[i] = -1 if g_i has a
             root and g_i is decreasing, and iroots[i] = 0 if g_i
             has no roots or g_i varies in the direction opposite
             to that indicated by rootdir[i].

  This routine returns an int equal to:
    ARK_RTFUNC_FAIL < 0 if the g function failed, or
    RTFOUND         = 1 if a root of g was found, or
    ARK_SUCCESS     = 0 otherwise.
  ---------------------------------------------------------------*/
pub fn arkRootfind(arkode_mem: &ARKodeMem) -> i32 {
    /* the C `arkode_mem == NULL` branch is unrepresentable */
    let ark_mem = arkode_mem;

    /* Move the rootfinding state into locals for the duration of the search
    (the user's g function is invoked inside the loop; no RefCell borrow may
    be held across it). C writes through the rootmem fields on every return
    path; the single write-back below restores the identical state for each
    path (on the ARK_RTFUNC_FAIL path the fields hold the values from the
    last completed iteration, exactly as in C). */
    let (nrtfn, ttol, tempv4) = {
        let m = ark_mem.borrow();
        let root_mem = m.root_mem.as_ref().expect("root_mem");
        (
            root_mem.nrtfn as usize,
            root_mem.ttol,
            m.tempv4.clone().expect("tempv4"),
        )
    };
    let (mut tlo, mut thi, mut trout) = {
        let m = ark_mem.borrow();
        let root_mem = m.root_mem.as_ref().expect("root_mem");
        (root_mem.tlo, root_mem.thi, root_mem.trout)
    };
    let (mut glo, mut ghi, mut grout, mut iroots, rootdir, gactive) = {
        let mut m = ark_mem.borrow_mut();
        let root_mem = m.root_mem.as_mut().expect("root_mem");
        (
            std::mem::take(&mut root_mem.glo),
            std::mem::take(&mut root_mem.ghi),
            std::mem::take(&mut root_mem.grout),
            std::mem::take(&mut root_mem.iroots),
            std::mem::take(&mut root_mem.rootdir),
            std::mem::take(&mut root_mem.gactive),
        )
    };

    let retflag = {
        let mut search = || -> i32 {
            let mut imax: usize = 0;

            /* First check for change in sign in ghi or for a zero in ghi. */
            let mut maxfrac = ZERO;
            let mut zroot = SUNFALSE;
            let mut sgnchg = SUNFALSE;
            for i in 0..nrtfn {
                if !gactive[i] {
                    continue;
                }
                if SUNRabs(ghi[i]) == ZERO {
                    if rootdir[i] as sunrealtype * glo[i] <= ZERO {
                        zroot = SUNTRUE;
                    }
                } else {
                    if SUNRdifferentsign(glo[i], ghi[i])
                        && (rootdir[i] as sunrealtype * glo[i] <= ZERO)
                    {
                        let gfrac = SUNRabs(ghi[i] / (ghi[i] - glo[i]));
                        if gfrac > maxfrac {
                            sgnchg = SUNTRUE;
                            maxfrac = gfrac;
                            imax = i;
                        }
                    }
                }
            }

            /* If no sign change was found, reset trout and grout.  Then return
               ARK_SUCCESS if no zero was found, or set iroots and return RTFOUND.  */
            if !sgnchg {
                trout = thi;
                for i in 0..nrtfn {
                    grout[i] = ghi[i];
                }
                if !zroot {
                    return ARK_SUCCESS;
                }
                for i in 0..nrtfn {
                    iroots[i] = 0;
                    if !gactive[i] {
                        continue;
                    }
                    if SUNRabs(ghi[i]) == ZERO {
                        iroots[i] = if glo[i] > ZERO { -1 } else { 1 };
                    }
                }
                return RTFOUND;
            }

            /* Initialize alpha to avoid compiler warning */
            let mut alpha = ONE;

            /* A sign change was found.  Loop to locate nearest root. */
            let mut side = 0;
            let mut sideprev = -1;
            loop {
                /* Looping point */

                /* If interval size is already less than tolerance ttol, break. */
                if SUNRabs(thi - tlo) <= ttol {
                    break;
                }

                /* Set weight alpha.
                   On the first two passes, set alpha = 1.  Thereafter, reset alpha
                   according to the side (low vs high) of the subinterval in which
                   the sign change was found in the previous two passes.
                   If the sides were opposite, set alpha = 1.
                   If the sides were the same, then double alpha (if high side),
                   or halve alpha (if low side).
                   The next guess tmid is the secant method value if alpha = 1, but
                   is closer to tlo if alpha < 1, and closer to thi if alpha > 1.    */
                if sideprev == side {
                    alpha = if side == 2 { alpha * TWO } else { alpha * HALF };
                } else {
                    alpha = ONE;
                }

                /* Set next root approximation tmid and get g(tmid).
                   If tmid is too close to tlo or thi, adjust it inward,
                   by a fractional distance that is between 0.1 and 0.5.  */
                let mut tmid = thi - (thi - tlo) * ghi[imax] / (ghi[imax] - alpha * glo[imax]);
                if SUNRabs(tmid - tlo) < HALF * ttol {
                    let fracint = SUNRabs(thi - tlo) / ttol;
                    let fracsub = if fracint > FIVE { TENTH } else { HALF / fracint };
                    tmid = tlo + fracsub * (thi - tlo);
                }
                if SUNRabs(thi - tmid) < HALF * ttol {
                    let fracint = SUNRabs(thi - tlo) / ttol;
                    let fracsub = if fracint > FIVE { TENTH } else { HALF / fracint };
                    tmid = thi - fracsub * (thi - tlo);
                }

                let _ = ARKodeGetDky(ark_mem, tmid, 0, &tempv4);
                let retval = ark_call_gfun(ark_mem, tmid, &tempv4, &mut grout);
                ark_mem
                    .borrow_mut()
                    .root_mem
                    .as_mut()
                    .expect("root_mem")
                    .nge += 1;
                if retval != 0 {
                    return ARK_RTFUNC_FAIL;
                }

                /* Check to see in which subinterval g changes sign, and reset imax.
                   Set side = 1 if sign change is on low side, or 2 if on high side.  */
                maxfrac = ZERO;
                zroot = SUNFALSE;
                sgnchg = SUNFALSE;
                sideprev = side;
                for i in 0..nrtfn {
                    if !gactive[i] {
                        continue;
                    }
                    if SUNRabs(grout[i]) == ZERO {
                        if rootdir[i] as sunrealtype * glo[i] <= ZERO {
                            zroot = SUNTRUE;
                        }
                    } else {
                        if SUNRdifferentsign(glo[i], grout[i])
                            && (rootdir[i] as sunrealtype * glo[i] <= ZERO)
                        {
                            let gfrac = SUNRabs(grout[i] / (grout[i] - glo[i]));
                            if gfrac > maxfrac {
                                sgnchg = SUNTRUE;
                                maxfrac = gfrac;
                                imax = i;
                            }
                        }
                    }
                }
                if sgnchg {
                    /* Sign change found in (tlo,tmid); replace thi with tmid. */
                    thi = tmid;
                    for i in 0..nrtfn {
                        ghi[i] = grout[i];
                    }
                    side = 1;
                    /* Stop at root thi if converged; otherwise loop. */
                    if SUNRabs(thi - tlo) <= ttol {
                        break;
                    }
                    continue; /* Return to looping point. */
                }

                if zroot {
                    /* No sign change in (tlo,tmid), but g = 0 at tmid; return root tmid. */
                    thi = tmid;
                    for i in 0..nrtfn {
                        ghi[i] = grout[i];
                    }
                    break;
                }

                /* No sign change in (tlo,tmid), and no zero at tmid.
                   Sign change must be in (tmid,thi).  Replace tlo with tmid. */
                tlo = tmid;
                for i in 0..nrtfn {
                    glo[i] = grout[i];
                }
                side = 2;
                /* Stop at root thi if converged; otherwise loop back. */
                if SUNRabs(thi - tlo) <= ttol {
                    break;
                }
            } /* End of root-search loop */

            /* Reset trout and grout, set iroots, and return RTFOUND. */
            trout = thi;
            for i in 0..nrtfn {
                grout[i] = ghi[i];
                iroots[i] = 0;
                if !gactive[i] {
                    continue;
                }
                if (SUNRabs(ghi[i]) == ZERO) && (rootdir[i] as sunrealtype * glo[i] <= ZERO) {
                    iroots[i] = if glo[i] > ZERO { -1 } else { 1 };
                }
                if SUNRdifferentsign(glo[i], ghi[i]) && (rootdir[i] as sunrealtype * glo[i] <= ZERO)
                {
                    iroots[i] = if glo[i] > ZERO { -1 } else { 1 };
                }
            }
            RTFOUND
        };
        search()
    };

    /* Write the rootfinding state back into the mem (single exit point) */
    {
        let mut m = ark_mem.borrow_mut();
        let root_mem = m.root_mem.as_mut().expect("root_mem");
        root_mem.tlo = tlo;
        root_mem.thi = thi;
        root_mem.trout = trout;
        root_mem.glo = glo;
        root_mem.ghi = ghi;
        root_mem.grout = grout;
        root_mem.iroots = iroots;
        root_mem.rootdir = rootdir;
        root_mem.gactive = gactive;
    }

    retflag
}

/*===============================================================
  EOF
  ===============================================================*/
