//! Port of `src/ida/ida.c` (+ `include/ida/ida.h` folded).
//!
//! Main IDA integrator: creation/initialization, tolerance functions,
//! rootfinding initialization, the `IDASolve` driver, `IDAStep` and all
//! its helpers, dense output (`IDAGetDky`, `IDAGetSolution`), the
//! internal error-weight functions, the WRMS norm helper, rootfinding
//! (`IDARcheck1/2/3`, `IDARootfind`), and `IDAFree`.
//!
//! `IDAProcessError`, the shared module-scope constants that upstream
//! repeats at the top of `ida.c` / `ida_ic.c` / `ida_io.c` (`ZERO`,
//! `ONE`, `PT001`, `ONEPSM`, `PREDICT_AGAIN`, `RTFOUND`, `IDA_SS`, …)
//! and every `MSG_*` message live in `ida_impl` (fragment protocol —
//! one shared definition, pulled in below by `use crate::ida_impl::*`).
//! Only the constants that are genuinely private to `ida.c` (the
//! "Algorithmic constants" block) are redefined here.
//!
//! Reference build configuration: SUNDIALS_LOGGING_LEVEL = 2
//! (SUNLogInfo/SUNLogInfoIf/SUNLogDebug/SUNLogExtraDebug* call sites
//! omitted; IDA_WARNING paths kept — they queue through the logger and
//! appear in the reference outputs), profiling off (all
//! `SUNDIALS_MARK_FUNCTION_BEGIN/END` omitted), error checks off,
//! monitoring on, serial branches only.
//!
//! Borrow discipline: internal functions take `&IDAMem` and use
//! granular borrows — no borrow of the mem is ever held across a user
//! callback, an N_Vector operation on a user vector, an `IDAProcessError`
//! call, or a linear/nonlinear solver call, all of which can re-enter
//! the mem.

use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

use crate::ida_impl::*;
use sundials_core::sundials_context::SUNContext;
use sundials_core::sundials_math::*;
use sundials_core::sundials_nonlinearsolver::*;
use sundials_core::sundials_nvector::*;
use sundials_core::sundials_types::*;
use sundials_core::sunnonlinsol_newton::SUNNonlinSol_Newton;

/*
 * =================================================================
 * IDA PRIVATE CONSTANTS
 * =================================================================
 *
 * The real constants (`ZERO`, `HALF`, `TWOTHIRDS`, `ONE`, `ONEPT5`,
 * `TWO`, `FOUR`, `FIVE`, `TEN`, `TWENTY`, `HUNDRED`, `PT9`, `PT1`,
 * `PT01`, `PT001`, `PT0001`, `ONEPSM`) and the routine-specific control
 * constants (`PREDICT_AGAIN`, `CONTINUE_STEPS`, `UNSET`, `LOWER`,
 * `RAISE`, `MAINTAIN`, `ERROR_TEST_FAIL`, `RTFOUND`, `CLOSERT`,
 * `IDA_NN`, `IDA_SS`, `IDA_SV`, `IDA_WF`) come from `ida_impl` — see
 * the fragment protocol note there. Only the block below is private to
 * `ida.c`.
 */

/*
 * Algorithmic constants
 * ---------------------
 */

const MXNCF: i32 = 10; /* max number of convergence failures allowed */
const MXNEF: i32 = 10; /* max number of error test failures allowed  */
const MAXNH: i32 = 5; /* max. number of h tries in IC calc. */
const MAXNJ: i32 = 4; /* max. number of J tries in IC calc. */
const MAXNI: i32 = 10; /* max. Newton iterations in IC calc. */
const EPCON: sunrealtype = 0.33; /* Newton convergence test constant */
const MAXBACKS: i32 = 100; /* max backtracks per Newton step in IDACalcIC */

/*
 * =================================================================
 * Callback invocation helpers (granular borrow discipline: the box
 * token is taken out of the mem around every user callback call and
 * restored afterwards on every path; no mem borrow is held across the
 * call)
 * =================================================================
 */

/// Invoke the error weight function
/// (C: `IDA_mem->ida_efun(ycur, weight, IDA_mem->ida_edata)`).
///
/// In C, `ida_edata` aliases `ida_user_data` when the user supplied
/// `efun` (`edata = user_data` in `IDAInitialSetup`) and points at
/// `IDA_mem` for the built-in `IDAEwtSet`. Box aliasing is impossible
/// in safe Rust, so user-efun call sites pass `ida_user_data` directly;
/// the observable behavior is identical.
fn ida_call_efun(IDA_mem: &IDAMem, ycur: &N_Vector, weight: &N_Vector) -> i32 {
    let (efun, user_efun) = {
        let m = IDA_mem.borrow();
        (m.ida_efun, m.ida_user_efun)
    };
    let efun = efun.expect("ida_efun set");
    if user_efun {
        let mut data = IDA_mem.borrow_mut().ida_user_data.take();
        let retval = efun(ycur, weight, &mut data);
        IDA_mem.borrow_mut().ida_user_data = data;
        retval
    } else {
        let mut data = IDA_mem.borrow_mut().ida_edata.take();
        let retval = efun(ycur, weight, &mut data);
        IDA_mem.borrow_mut().ida_edata = data;
        retval
    }
}

/* The root function `g` is invoked inline at its six call sites
(IDARcheck1/2/3, IDARootfind): each one also needs `std::mem::take` of the
`ida_glo`/`ida_ghi`/`ida_grout` mem field that receives `gout`, which a
shared `&mut [sunrealtype]` helper cannot express. */

/*
 * =================================================================
 * EXPORTED FUNCTIONS IMPLEMENTATION
 * =================================================================
 */

/*
 * -----------------------------------------------------------------
 * Creation, allocation and re-initialization functions
 * -----------------------------------------------------------------
 */

/*
 * IDACreate
 *
 * IDACreate creates an internal memory block for a problem to
 * be solved by IDA.
 * If successful, IDACreate returns a pointer to the problem memory.
 * This pointer should be passed to IDAInit.
 * If an initialization error occurs, IDACreate prints an error
 * message to standard err and returns NULL.
 */

pub fn IDACreate(sunctx: &SUNContext) -> Option<IDAMem> {
    /* Test inputs */
    /* NULL sunctx check: handled by type system */

    /* malloc failure branch: allocation cannot fail observably in Rust */
    /* Zero out ida_mem: IDAMemRec::zeroed is the memset */
    let mut IDA_mem = IDAMemRec::zeroed(sunctx.clone());

    /* IDA_mem->ida_sunctx = sunctx (set by zeroed) */

    /* Set unit roundoff in IDA_mem */
    IDA_mem.ida_uround = SUN_UNIT_ROUNDOFF;

    /* Set default values for integrator optional inputs */
    IDA_mem.ida_res = None;
    IDA_mem.ida_user_data = None;
    IDA_mem.ida_itol = IDA_NN;
    IDA_mem.ida_atolmin0 = SUNTRUE;
    IDA_mem.ida_user_efun = SUNFALSE;
    IDA_mem.ida_efun = None;
    IDA_mem.ida_edata = None;
    IDA_mem.ida_maxord = MAXORD_DEFAULT as i32;
    IDA_mem.ida_mxstep = MXSTEP_DEFAULT;
    IDA_mem.ida_hmax_inv = HMAX_INV_DEFAULT;
    IDA_mem.ida_hmin = HMIN_DEFAULT;
    IDA_mem.ida_eta_max_fx = ETA_MAX_FX_DEFAULT;
    IDA_mem.ida_eta_min_fx = ETA_MIN_FX_DEFAULT;
    IDA_mem.ida_eta_max = ETA_MAX_DEFAULT;
    IDA_mem.ida_eta_low = ETA_LOW_DEFAULT;
    IDA_mem.ida_eta_min = ETA_MIN_DEFAULT;
    IDA_mem.ida_eta_min_ef = ETA_MIN_EF_DEFAULT;
    IDA_mem.ida_eta_cf = ETA_CF_DEFAULT;
    IDA_mem.ida_hin = ZERO;
    IDA_mem.ida_epcon = EPCON;
    IDA_mem.ida_maxnef = MXNEF;
    IDA_mem.ida_maxncf = MXNCF;
    IDA_mem.ida_suppressalg = SUNFALSE;
    IDA_mem.ida_id = None;
    IDA_mem.ida_tstopset = SUNFALSE;
    IDA_mem.ida_dcj = DCJ_DEFAULT;

    /* Initialize inequality constraint variables */
    IDA_mem.ida_constraints = None;
    IDA_mem.constraint_corrections = 0;
    IDA_mem.constraint_fails = 0;
    IDA_mem.max_constraint_fails = MAX_CONSTRAINT_FAILS;

    /* set the saved value maxord_alloc */
    IDA_mem.ida_maxord_alloc = MAXORD_DEFAULT as i32;

    /* Set default values for IC optional inputs */
    IDA_mem.ida_epiccon = PT01 * EPCON;
    IDA_mem.ida_maxnh = MAXNH;
    IDA_mem.ida_maxnj = MAXNJ;
    IDA_mem.ida_maxnit = MAXNI;
    IDA_mem.ida_maxbacks = MAXBACKS;
    IDA_mem.ida_lsoff = SUNFALSE;
    IDA_mem.ida_steptol = SUNRpowerR(IDA_mem.ida_uround, TWOTHIRDS);

    /* Initialize lrw and liw */
    IDA_mem.ida_lrw = (25 + 5 * MXORDP1) as i64;
    IDA_mem.ida_liw = 38;

    /* No mallocs have been done yet */
    IDA_mem.ida_VatolMallocDone = SUNFALSE;
    IDA_mem.ida_idMallocDone = SUNFALSE;
    IDA_mem.ida_MallocDone = SUNFALSE;

    /* Initialize nonlinear solver variables */
    IDA_mem.NLS = None;
    IDA_mem.ownNLS = SUNFALSE;

    /* Return pointer to IDA memory block */
    Some(Rc::new(RefCell::new(IDA_mem)))
}

/*-----------------------------------------------------------------*/

/*
 * IDAInit
 *
 * IDAInit allocates and initializes memory for a problem. All
 * problem specification inputs are checked for errors. If any
 * error occurs during initialization, it is reported to the
 * error handler function.
 */

pub fn IDAInit(
    ida_mem: &IDAMem,
    res: IDAResFn,
    t0: sunrealtype,
    yy0: &N_Vector,
    yp0: &N_Vector,
) -> i32 {
    /* Check ida_mem */
    /* NULL-mem check: handled by type system */
    let IDA_mem = ida_mem;

    /* Check for legal input parameters */

    /* NULL yy0 check: handled by type system */
    /* NULL yp0 check: handled by type system */
    /* NULL res check: handled by type system */

    /* Test if all required vector operations are implemented */

    let nvectorOK = IDACheckNvector(yy0);
    if !nvectorOK {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDAInit",
            file!(),
            MSG_BAD_NVECTOR,
        );
        return IDA_ILL_INPUT;
    }

    /* Set space requirements for one N_Vector */

    let mut lrw1: sunindextype = 0;
    let mut liw1: sunindextype = 0;
    if yy0.ops.borrow().nvspace.is_some() {
        N_VSpace(yy0, &mut lrw1, &mut liw1);
    } else {
        lrw1 = 0;
        liw1 = 0;
    }
    {
        let mut m = IDA_mem.borrow_mut();
        m.ida_lrw1 = lrw1;
        m.ida_liw1 = liw1;
    }

    /* Allocate the vectors (using yy0 as a template) */

    let allocOK = IDAAllocVectors(IDA_mem, yy0);
    if !allocOK {
        IDAProcessError(
            Some(IDA_mem),
            IDA_MEM_FAIL,
            line!() as i32,
            "IDAInit",
            file!(),
            MSG_MEM_FAIL,
        );
        return IDA_MEM_FAIL;
    }

    /* Input checks complete at this point and history array allocated */

    /* Copy the input parameters into IDA memory block */
    {
        let mut m = IDA_mem.borrow_mut();
        m.ida_res = Some(res);
        m.ida_tn = t0;
    }

    /* Initialize the phi array */
    let (phi0, phi1) = {
        let m = IDA_mem.borrow();
        (m.ida_phi[0].clone().unwrap(), m.ida_phi[1].clone().unwrap())
    };
    N_VScale(ONE, yy0, &phi0);
    N_VScale(ONE, yp0, &phi1);

    /* create a Newton nonlinear solver object by default */
    let sunctx = IDA_mem.borrow().ida_sunctx.clone();
    let NLS = SUNNonlinSol_Newton(yy0, &sunctx);

    /* check that nonlinear solver is non-NULL */
    let NLS = match NLS {
        Some(nls) => nls,
        None => {
            IDAProcessError(
                Some(IDA_mem),
                IDA_MEM_FAIL,
                line!() as i32,
                "IDAInit",
                file!(),
                MSG_MEM_FAIL,
            );
            IDAFreeVectors(IDA_mem);
            return IDA_MEM_FAIL;
        }
    };

    /* attach the nonlinear solver to the IDA memory */
    let retval = crate::ida_nls::IDASetNonlinearSolver(IDA_mem, &NLS);

    /* check that the nonlinear solver was successfully attached */
    if retval != IDA_SUCCESS {
        IDAProcessError(
            Some(IDA_mem),
            retval,
            line!() as i32,
            "IDAInit",
            file!(),
            "Setting the nonlinear solver failed",
        );
        IDAFreeVectors(IDA_mem);
        let _ = SUNNonlinSolFree(Some(NLS));
        return IDA_MEM_FAIL;
    }

    /* set ownership flag */
    IDA_mem.borrow_mut().ownNLS = SUNTRUE;

    /* All error checking is complete at this point */

    {
        let mut m = IDA_mem.borrow_mut();

        /* Set the linear solver addresses to NULL */

        m.ida_linit = None;
        m.ida_lsetup = None;
        m.ida_lsolve = None;
        m.ida_lperf = None;
        m.ida_lfree = None;
        m.ida_lmem = None;

        /* Initialize all the counters and other optional output values */

        m.ida_nst = 0;
        m.ida_nre = 0;
        m.ida_ncfn = 0;
        m.ida_netf = 0;
        m.ida_nni = 0;
        m.ida_nnf = 0;
        m.ida_nsetups = 0;

        m.ida_kused = 0;
        m.ida_hused = ZERO;
        m.ida_tolsf = ONE;

        m.ida_nge = 0;

        m.ida_irfnd = 0;

        /* Initialize counters specific to IC calculation. */
        m.ida_nbacktr = 0;

        /* Initialize root-finding variables */

        m.ida_glo = Vec::new();
        m.ida_ghi = Vec::new();
        m.ida_grout = Vec::new();
        m.ida_iroots = Vec::new();
        m.ida_rootdir = Vec::new();
        m.ida_gfun = None;
        m.ida_nrtfn = 0;
        m.ida_gactive = Vec::new();
        m.ida_mxgnull = 1;

        /* Initial setup not done yet */

        m.ida_SetupDone = SUNFALSE;

        /* Problem memory has been successfully allocated */

        m.ida_MallocDone = SUNTRUE;
    }

    IDA_SUCCESS
}

/*-----------------------------------------------------------------*/

/*
 * IDAReInit
 *
 * IDAReInit re-initializes IDA's memory for a problem, assuming
 * it has already been allocated in a prior IDAInit call.
 * All problem specification inputs are checked for errors.
 * The problem size Neq is assumed to be unchanged since the call
 * to IDAInit, and the maximum order maxord must not be larger.
 * If any error occurs during reinitialization, it is reported to
 * the error handler function.
 * The return value is IDA_SUCCESS = 0 if no errors occurred, or
 * a negative value otherwise.
 */

pub fn IDAReInit(ida_mem: &IDAMem, t0: sunrealtype, yy0: &N_Vector, yp0: &N_Vector) -> i32 {
    /* Check for legal input parameters */

    /* NULL-mem check: handled by type system */
    let IDA_mem = ida_mem;

    /* Check if problem was malloc'ed */

    if !IDA_mem.borrow().ida_MallocDone {
        IDAProcessError(
            Some(IDA_mem),
            IDA_NO_MALLOC,
            line!() as i32,
            "IDAReInit",
            file!(),
            MSG_NO_MALLOC,
        );
        return IDA_NO_MALLOC;
    }

    /* Check for legal input parameters */

    /* NULL yy0 check: handled by type system */
    /* NULL yp0 check: handled by type system */

    /* Copy the input parameters into IDA memory block */

    IDA_mem.borrow_mut().ida_tn = t0;

    /* Initialize the phi array */

    let (phi0, phi1) = {
        let m = IDA_mem.borrow();
        (m.ida_phi[0].clone().unwrap(), m.ida_phi[1].clone().unwrap())
    };
    N_VScale(ONE, yy0, &phi0);
    N_VScale(ONE, yp0, &phi1);

    {
        let mut m = IDA_mem.borrow_mut();

        /* Initialize all the counters and other optional output values */

        m.ida_nst = 0;
        m.ida_nre = 0;
        m.ida_ncfn = 0;
        m.ida_netf = 0;
        m.ida_nni = 0;
        m.ida_nnf = 0;
        m.ida_nsetups = 0;

        m.ida_kused = 0;
        m.ida_hused = ZERO;
        m.ida_tolsf = ONE;

        m.ida_nge = 0;

        m.ida_irfnd = 0;

        m.constraint_corrections = 0;
        m.constraint_fails = 0;
    }

    if IDA_mem.borrow().ida_lmem.is_some() {
        let _ = crate::ida_ls::idaLsInitializeCounters(&mut crate::ida_ls::idals_mem_mut(IDA_mem));
    }

    /* Initial setup not done yet */

    IDA_mem.borrow_mut().ida_SetupDone = SUNFALSE;

    /* Problem has been successfully re-initialized */

    IDA_SUCCESS
}

/*-----------------------------------------------------------------*/

/*
 * IDASStolerances
 * IDASVtolerances
 * IDAWFtolerances
 *
 * These functions specify the integration tolerances. One of them
 * MUST be called before the first call to IDA.
 *
 * IDASStolerances specifies scalar relative and absolute tolerances.
 * IDASVtolerances specifies scalar relative tolerance and a vector
 *   absolute tolerance (a potentially different absolute tolerance
 *   for each vector component).
 * IDAWFtolerances specifies a user-provides function (of type IDAEwtFn)
 *   which will be called to set the error weight vector.
 */

pub fn IDASStolerances(ida_mem: &IDAMem, reltol: sunrealtype, abstol: sunrealtype) -> i32 {
    /* NULL-mem check: handled by type system */
    let IDA_mem = ida_mem;

    if !IDA_mem.borrow().ida_MallocDone {
        IDAProcessError(
            Some(IDA_mem),
            IDA_NO_MALLOC,
            line!() as i32,
            "IDASStolerances",
            file!(),
            MSG_NO_MALLOC,
        );
        return IDA_NO_MALLOC;
    }

    /* Check inputs */

    if reltol < ZERO {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDASStolerances",
            file!(),
            MSG_BAD_RTOL,
        );
        return IDA_ILL_INPUT;
    }

    if abstol < ZERO {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDASStolerances",
            file!(),
            MSG_BAD_ATOL,
        );
        return IDA_ILL_INPUT;
    }

    /* Copy tolerances into memory */

    let mut m = IDA_mem.borrow_mut();
    m.ida_rtol = reltol;
    m.ida_Satol = abstol;
    m.ida_atolmin0 = abstol == ZERO;

    m.ida_itol = IDA_SS;

    m.ida_user_efun = SUNFALSE;
    m.ida_efun = Some(IDAEwtSet);
    m.ida_edata = None; /* will be set to ida_mem in InitialSetup */

    IDA_SUCCESS
}

pub fn IDASVtolerances(ida_mem: &IDAMem, reltol: sunrealtype, abstol: &N_Vector) -> i32 {
    /* NULL-mem check: handled by type system */
    let IDA_mem = ida_mem;

    if !IDA_mem.borrow().ida_MallocDone {
        IDAProcessError(
            Some(IDA_mem),
            IDA_NO_MALLOC,
            line!() as i32,
            "IDASVtolerances",
            file!(),
            MSG_NO_MALLOC,
        );
        return IDA_NO_MALLOC;
    }

    /* Check inputs */

    if reltol < ZERO {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDASVtolerances",
            file!(),
            MSG_BAD_RTOL,
        );
        return IDA_ILL_INPUT;
    }

    let atolmin = N_VMin(abstol);
    if atolmin < ZERO {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDASVtolerances",
            file!(),
            MSG_BAD_ATOL,
        );
        return IDA_ILL_INPUT;
    }

    /* Copy tolerances into memory */

    if !IDA_mem.borrow().ida_VatolMallocDone {
        let ewt = IDA_mem.borrow().ida_ewt.clone().unwrap();
        let vatol = N_VClone(&ewt).unwrap();
        let mut m = IDA_mem.borrow_mut();
        m.ida_Vatol = Some(vatol);
        m.ida_lrw += m.ida_lrw1;
        m.ida_liw += m.ida_liw1;
        m.ida_VatolMallocDone = SUNTRUE;
    }

    IDA_mem.borrow_mut().ida_rtol = reltol;
    let vatol = IDA_mem.borrow().ida_Vatol.clone().unwrap();
    N_VScale(ONE, abstol, &vatol);
    {
        let mut m = IDA_mem.borrow_mut();
        m.ida_atolmin0 = atolmin == ZERO;

        m.ida_itol = IDA_SV;

        m.ida_user_efun = SUNFALSE;
        m.ida_efun = Some(IDAEwtSet);
        m.ida_edata = None; /* will be set to ida_mem in InitialSetup */
    }

    IDA_SUCCESS
}

pub fn IDAWFtolerances(ida_mem: &IDAMem, efun: IDAEwtFn) -> i32 {
    /* NULL-mem check: handled by type system */
    let IDA_mem = ida_mem;

    if !IDA_mem.borrow().ida_MallocDone {
        IDAProcessError(
            Some(IDA_mem),
            IDA_NO_MALLOC,
            line!() as i32,
            "IDAWFtolerances",
            file!(),
            MSG_NO_MALLOC,
        );
        return IDA_NO_MALLOC;
    }

    let mut m = IDA_mem.borrow_mut();
    m.ida_itol = IDA_WF;

    m.ida_user_efun = SUNTRUE;
    m.ida_efun = Some(efun);
    m.ida_edata = None; /* will be set to user_data in InitialSetup */

    IDA_SUCCESS
}

/*-----------------------------------------------------------------*/

/*
 * IDARootInit
 *
 * IDARootInit initializes a rootfinding problem to be solved
 * during the integration of the DAE system.  It loads the root
 * function pointer and the number of root functions, and allocates
 * workspace memory.  The return value is IDA_SUCCESS = 0 if no
 * errors occurred, or a negative value otherwise.
 */

pub fn IDARootInit(ida_mem: &IDAMem, nrtfn: i32, g: Option<IDARootFn>) -> i32 {
    /* Check ida_mem pointer */
    /* NULL-mem check: handled by type system */
    let IDA_mem = ida_mem;

    let nrt = if nrtfn < 0 { 0 } else { nrtfn };

    /* If rerunning IDARootInit() with a different number of root
    functions (changing number of gfun components), then free
    currently held memory resources */
    {
        let mut m = IDA_mem.borrow_mut();
        if (nrt != m.ida_nrtfn) && (m.ida_nrtfn > 0) {
            m.ida_glo = Vec::new();
            m.ida_ghi = Vec::new();
            m.ida_grout = Vec::new();
            m.ida_iroots = Vec::new();
            m.ida_rootdir = Vec::new();
            m.ida_gactive = Vec::new();

            m.ida_lrw -= 3 * (m.ida_nrtfn as i64);
            m.ida_liw -= 3 * (m.ida_nrtfn as i64);
        }
    }

    /* If IDARootInit() was called with nrtfn == 0, then set ida_nrtfn to
    zero and ida_gfun to NULL before returning */
    if nrt == 0 {
        let mut m = IDA_mem.borrow_mut();
        m.ida_nrtfn = nrt;
        m.ida_gfun = None;
        return IDA_SUCCESS;
    }

    /* If rerunning IDARootInit() with the same number of root functions
    (not changing number of gfun components), then check if the root
    function argument has changed */
    /* If g != NULL then return as currently reserved memory resources
    will suffice */
    if nrt == IDA_mem.borrow().ida_nrtfn {
        let mut m = IDA_mem.borrow_mut();
        /* C compares the root-fn pointers by identity; fn-pointer identity
        in Rust carries the same caveats as C across translation units */
        #[allow(unpredictable_function_pointer_comparisons)]
        if g != m.ida_gfun {
            if g.is_none() {
                m.ida_glo = Vec::new();
                m.ida_ghi = Vec::new();
                m.ida_grout = Vec::new();
                m.ida_iroots = Vec::new();
                m.ida_rootdir = Vec::new();
                m.ida_gactive = Vec::new();

                m.ida_lrw -= 3 * (nrt as i64);
                m.ida_liw -= 3 * (nrt as i64);

                drop(m);
                IDAProcessError(
                    Some(IDA_mem),
                    IDA_ILL_INPUT,
                    line!() as i32,
                    "IDARootInit",
                    file!(),
                    MSG_ROOT_FUNC_NULL,
                );
                return IDA_ILL_INPUT;
            } else {
                m.ida_gfun = g;
                return IDA_SUCCESS;
            }
        } else {
            return IDA_SUCCESS;
        }
    }

    /* Set variable values in IDA memory block */
    IDA_mem.borrow_mut().ida_nrtfn = nrt;
    if g.is_none() {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDARootInit",
            file!(),
            MSG_ROOT_FUNC_NULL,
        );
        return IDA_ILL_INPUT;
    } else {
        IDA_mem.borrow_mut().ida_gfun = g;
    }

    /* Allocate necessary memory and return (the C allocation-failure
    branches are unreachable in Rust: Vec allocation aborts rather than
    returning NULL) */
    {
        let mut m = IDA_mem.borrow_mut();
        m.ida_glo = vec![ZERO; nrt as usize];
        m.ida_ghi = vec![ZERO; nrt as usize];
        m.ida_grout = vec![ZERO; nrt as usize];
        m.ida_iroots = vec![0; nrt as usize];

        /* Set default values for rootdir (both directions) */
        m.ida_rootdir = vec![0; nrt as usize];

        /* Set default values for gactive (all active) */
        m.ida_gactive = vec![SUNTRUE; nrt as usize];

        m.ida_lrw += 3 * (nrt as i64);
        m.ida_liw += 3 * (nrt as i64);
    }

    IDA_SUCCESS
}

/*
 * -----------------------------------------------------------------
 * Main solver function
 * -----------------------------------------------------------------
 */

/*
 * IDASolve
 *
 * This routine is the main driver of the IDA package.
 *
 * It integrates over an independent variable interval defined by the user,
 * by calling IDAStep to take internal independent variable steps.
 *
 * The first time that IDASolve is called for a successfully initialized
 * problem, it computes a tentative initial step size.
 *
 * IDASolve supports two modes, specified by itask:
 * In the IDA_NORMAL mode, the solver steps until it passes tout and then
 * interpolates to obtain y(tout) and yp(tout).
 * In the IDA_ONE_STEP mode, it takes one internal step and returns.
 *
 * IDASolve returns integer values corresponding to success and failure as below:
 *
 * successful returns:
 *
 * IDA_SUCCESS
 * IDA_TSTOP_RETURN
 *
 * failed returns:
 *
 * IDA_ILL_INPUT
 * IDA_TOO_MUCH_WORK
 * IDA_MEM_NULL
 * IDA_TOO_MUCH_ACC
 * IDA_CONV_FAIL
 * IDA_LSETUP_FAIL
 * IDA_LSOLVE_FAIL
 * IDA_CONSTR_FAIL
 * IDA_ERR_FAIL
 * IDA_REP_RES_ERR
 * IDA_RES_FAIL
 */

pub fn IDASolve(
    ida_mem: &IDAMem,
    tout: sunrealtype,
    tret: &mut sunrealtype,
    yret: &N_Vector,
    ypret: &N_Vector,
    itask: i32,
) -> i32 {
    /* Check for legal inputs in all cases. */

    /* NULL-mem check: handled by type system */
    let IDA_mem = ida_mem;

    /* Check if problem was malloc'ed */

    if !IDA_mem.borrow().ida_MallocDone {
        IDAProcessError(
            Some(IDA_mem),
            IDA_NO_MALLOC,
            line!() as i32,
            "IDASolve",
            file!(),
            MSG_NO_MALLOC,
        );
        return IDA_NO_MALLOC;
    }

    /* Check for legal arguments */

    /* NULL yret check: handled by type system. ida_yy aliases the user's
    yret (the Rc clone shares the underlying data exactly as the C
    pointer copy does), so the copy-back contract is satisfied by
    construction on every return path. */
    IDA_mem.borrow_mut().ida_yy = Some(yret.clone());

    /* NULL ypret check: handled by type system */
    IDA_mem.borrow_mut().ida_yp = Some(ypret.clone());

    /* NULL tret check: handled by type system */

    if (itask != IDA_NORMAL) && (itask != IDA_ONE_STEP) {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDASolve",
            file!(),
            MSG_BAD_ITASK,
        );
        return IDA_ILL_INPUT;
    }

    if IDA_mem.borrow().ida_nst == 0 {
        /* This is the first call */

        /* Check inputs to IDA for correctness and consistency */

        if !IDA_mem.borrow().ida_SetupDone {
            let ier = IDAInitialSetup(IDA_mem);
            if ier != IDA_SUCCESS {
                return ier;
            }
            IDA_mem.borrow_mut().ida_SetupDone = SUNTRUE;
        }

        /* On first call, check for tout - tn too small, set initial hh,
        check for approach to tstop, and scale phi[1] by hh.
        Also check for zeros of root function g at and near t0.    */

        let (tn, uround) = {
            let m = IDA_mem.borrow();
            (m.ida_tn, m.ida_uround)
        };
        let tdist = SUNRabs(tout - tn);
        let troundoff = TWO * uround * (SUNRabs(tn) + SUNRabs(tout));
        if tdist == ZERO || tdist < troundoff {
            IDAProcessError(
                Some(IDA_mem),
                IDA_TOO_CLOSE,
                line!() as i32,
                "IDASolve",
                file!(),
                MSG_TOO_CLOSE,
            );
            return IDA_TOO_CLOSE;
        }

        /* Set initial h */

        {
            let mut m = IDA_mem.borrow_mut();
            m.ida_hh = m.ida_hin;
            if (m.ida_hh != ZERO) && ((tout - m.ida_tn) * m.ida_hh < ZERO) {
                drop(m);
                IDAProcessError(
                    Some(IDA_mem),
                    IDA_ILL_INPUT,
                    line!() as i32,
                    "IDASolve",
                    file!(),
                    MSG_BAD_HINIT,
                );
                return IDA_ILL_INPUT;
            }
        }

        if IDA_mem.borrow().ida_hh == ZERO {
            IDA_mem.borrow_mut().ida_hh = PT001 * tdist;
            let (phi1, ewt, suppressalg) = {
                let m = IDA_mem.borrow();
                (
                    m.ida_phi[1].clone().unwrap(),
                    m.ida_ewt.clone().unwrap(),
                    m.ida_suppressalg,
                )
            };
            let ypnorm = IDAWrmsNorm(IDA_mem, &phi1, &ewt, suppressalg);
            let mut m = IDA_mem.borrow_mut();
            if ypnorm > HALF / m.ida_hh {
                m.ida_hh = HALF / ypnorm;
            }
            if tout < m.ida_tn {
                m.ida_hh = -m.ida_hh;
            }
        }

        /* Enforce hmax and hmin */

        {
            let mut m = IDA_mem.borrow_mut();
            let rh = SUNRabs(m.ida_hh) * m.ida_hmax_inv;
            if rh > ONE {
                m.ida_hh /= rh;
            }
            if SUNRabs(m.ida_hh) < m.ida_hmin {
                m.ida_hh *= m.ida_hmin / SUNRabs(m.ida_hh);
            }
        }

        /* Check for approach to tstop */

        if IDA_mem.borrow().ida_tstopset {
            {
                let m = IDA_mem.borrow();
                if (m.ida_tstop - m.ida_tn) * m.ida_hh <= ZERO {
                    let (tstop, tn) = (m.ida_tstop, m.ida_tn);
                    drop(m);
                    IDAProcessError(
                        Some(IDA_mem),
                        IDA_ILL_INPUT,
                        line!() as i32,
                        "IDASolve",
                        file!(),
                        &MSG_BAD_TSTOP(tstop, tn),
                    );
                    return IDA_ILL_INPUT;
                }
            }
            let mut m = IDA_mem.borrow_mut();
            if (m.ida_tn + m.ida_hh - m.ida_tstop) * m.ida_hh > ZERO {
                m.ida_hh = (m.ida_tstop - m.ida_tn) * (ONE - FOUR * m.ida_uround);
            }
        }

        {
            let mut m = IDA_mem.borrow_mut();
            m.ida_h0u = m.ida_hh;
            m.ida_kk = 0;
            m.ida_kused = 0; /* set in case of an error return before a step */
        }

        /* Check for exact zeros of the root functions at or near t0. */
        if IDA_mem.borrow().ida_nrtfn > 0 {
            let ier = IDARcheck1(IDA_mem);
            if ier == IDA_RTFUNC_FAIL {
                let tn = IDA_mem.borrow().ida_tn;
                IDAProcessError(
                    Some(IDA_mem),
                    IDA_RTFUNC_FAIL,
                    line!() as i32,
                    "IDASolve",
                    file!(),
                    &MSG_RTFUNC_FAILED(tn),
                );
                return IDA_RTFUNC_FAIL;
            }
        }

        /* set phi[1] = hh*y' */
        let (hh, phi1) = {
            let m = IDA_mem.borrow();
            (m.ida_hh, m.ida_phi[1].clone().unwrap())
        };
        N_VScale(hh, &phi1, &phi1);

        /* Set the convergence test constants epsNewt and toldel */
        {
            let mut m = IDA_mem.borrow_mut();
            m.ida_epsNewt = m.ida_epcon;
            m.ida_toldel = PT0001 * m.ida_epsNewt;
        }
    } /* end of first-call block. */

    /* Call lperf function and set nstloc for later performance testing. */

    let lperf = IDA_mem.borrow().ida_lperf;
    if let Some(lperf) = lperf {
        let _ = lperf(IDA_mem, 0);
    }
    let mut nstloc: i64 = 0;

    /* If not the first call, perform all stopping tests. */

    if IDA_mem.borrow().ida_nst > 0 {
        /* First, check for a root in the last step taken, other than the
        last root found, if any.  If itask = IDA_ONE_STEP and y(tn) was not
        returned because of an intervening root, return y(tn) now.     */

        if IDA_mem.borrow().ida_nrtfn > 0 {
            let irfndp = IDA_mem.borrow().ida_irfnd;

            let ier = IDARcheck2(IDA_mem);

            if ier == CLOSERT {
                let tlo = IDA_mem.borrow().ida_tlo;
                IDAProcessError(
                    Some(IDA_mem),
                    IDA_ILL_INPUT,
                    line!() as i32,
                    "IDASolve",
                    file!(),
                    &MSG_CLOSE_ROOTS(tlo),
                );
                return IDA_ILL_INPUT;
            } else if ier == IDA_RTFUNC_FAIL {
                let tlo = IDA_mem.borrow().ida_tlo;
                IDAProcessError(
                    Some(IDA_mem),
                    IDA_RTFUNC_FAIL,
                    line!() as i32,
                    "IDASolve",
                    file!(),
                    &MSG_RTFUNC_FAILED(tlo),
                );
                return IDA_RTFUNC_FAIL;
            } else if ier == RTFOUND {
                let tlo = IDA_mem.borrow().ida_tlo;
                IDA_mem.borrow_mut().ida_tretlast = tlo;
                *tret = tlo;
                return IDA_ROOT_RETURN;
            }

            /* If tn is distinct from tretlast (within roundoff),
            check remaining interval for roots */
            let distinct = {
                let m = IDA_mem.borrow();
                let troundoff = HUNDRED * m.ida_uround * (SUNRabs(m.ida_tn) + SUNRabs(m.ida_hh));
                SUNRabs(m.ida_tn - m.ida_tretlast) > troundoff
            };
            if distinct {
                let ier = IDARcheck3(IDA_mem, tout, itask);
                if ier == IDA_SUCCESS {
                    /* no root found */
                    IDA_mem.borrow_mut().ida_irfnd = 0;
                    if (irfndp == 1) && (itask == IDA_ONE_STEP) {
                        let tn = IDA_mem.borrow().ida_tn;
                        IDA_mem.borrow_mut().ida_tretlast = tn;
                        *tret = tn;
                        let _ = IDAGetSolution(IDA_mem, tn, yret, ypret);
                        return IDA_SUCCESS;
                    }
                } else if ier == RTFOUND {
                    /* a new root was found */
                    let tlo = IDA_mem.borrow().ida_tlo;
                    {
                        let mut m = IDA_mem.borrow_mut();
                        m.ida_irfnd = 1;
                        m.ida_tretlast = tlo;
                    }
                    *tret = tlo;
                    return IDA_ROOT_RETURN;
                } else if ier == IDA_RTFUNC_FAIL {
                    /* g failed */
                    let tlo = IDA_mem.borrow().ida_tlo;
                    IDAProcessError(
                        Some(IDA_mem),
                        IDA_RTFUNC_FAIL,
                        line!() as i32,
                        "IDASolve",
                        file!(),
                        &MSG_RTFUNC_FAILED(tlo),
                    );
                    return IDA_RTFUNC_FAIL;
                }
            }
        } /* end of root stop check */

        /* Now test for all other stop conditions. */

        let istate = IDAStopTest1(IDA_mem, tout, tret, yret, ypret, itask);
        if istate != CONTINUE_STEPS {
            return istate;
        }
    }

    /* Looping point for internal steps. */

    let mut istate: i32;
    loop {
        /* Check for too many steps taken. */

        {
            let (mxstep, tn) = {
                let m = IDA_mem.borrow();
                (m.ida_mxstep, m.ida_tn)
            };
            if (mxstep > 0) && (nstloc >= mxstep) {
                IDAProcessError(
                    Some(IDA_mem),
                    IDA_ILL_INPUT,
                    line!() as i32,
                    "IDASolve",
                    file!(),
                    &MSG_MAX_STEPS(tn),
                );
                istate = IDA_TOO_MUCH_WORK;
                IDA_mem.borrow_mut().ida_tretlast = tn;
                *tret = tn;
                break; /* Here yy=yret and yp=ypret already have the current solution. */
            }
        }

        /* Call lperf to generate warnings of poor performance. */

        let lperf = IDA_mem.borrow().ida_lperf;
        if let Some(lperf) = lperf {
            let _ = lperf(IDA_mem, 1);
        }

        /* Reset and check ewt (if not first call). */

        if IDA_mem.borrow().ida_nst > 0 {
            let (phi0, ewt) = {
                let m = IDA_mem.borrow();
                (m.ida_phi[0].clone().unwrap(), m.ida_ewt.clone().unwrap())
            };
            let ier = ida_call_efun(IDA_mem, &phi0, &ewt);

            if ier != 0 {
                let (itol, tn) = {
                    let m = IDA_mem.borrow();
                    (m.ida_itol, m.ida_tn)
                };
                if itol == IDA_WF {
                    IDAProcessError(
                        Some(IDA_mem),
                        IDA_ILL_INPUT,
                        line!() as i32,
                        "IDASolve",
                        file!(),
                        &MSG_EWT_NOW_FAIL(tn),
                    );
                } else {
                    IDAProcessError(
                        Some(IDA_mem),
                        IDA_ILL_INPUT,
                        line!() as i32,
                        "IDASolve",
                        file!(),
                        &MSG_EWT_NOW_BAD(tn),
                    );
                }

                istate = IDA_ILL_INPUT;
                let _ = IDAGetSolution(IDA_mem, tn, yret, ypret);
                IDA_mem.borrow_mut().ida_tretlast = tn;
                *tret = tn;
                break;
            }
        }

        /* Check for too much accuracy requested. */

        {
            let (phi0, ewt, suppressalg, uround) = {
                let m = IDA_mem.borrow();
                (
                    m.ida_phi[0].clone().unwrap(),
                    m.ida_ewt.clone().unwrap(),
                    m.ida_suppressalg,
                    m.ida_uround,
                )
            };
            let nrm = IDAWrmsNorm(IDA_mem, &phi0, &ewt, suppressalg);
            IDA_mem.borrow_mut().ida_tolsf = uround * nrm;
            if IDA_mem.borrow().ida_tolsf > ONE {
                let tn = {
                    let mut m = IDA_mem.borrow_mut();
                    m.ida_tolsf *= TEN;
                    m.ida_tn
                };
                IDAProcessError(
                    Some(IDA_mem),
                    IDA_ILL_INPUT,
                    line!() as i32,
                    "IDASolve",
                    file!(),
                    &MSG_TOO_MUCH_ACC(tn),
                );
                istate = IDA_TOO_MUCH_ACC;
                IDA_mem.borrow_mut().ida_tretlast = tn;
                *tret = tn;
                if IDA_mem.borrow().ida_nst > 0 {
                    let _ = IDAGetSolution(IDA_mem, tn, yret, ypret);
                }
                break;
            }
        }

        /* Call IDAStep to take a step. */

        let sflag = IDAStep(IDA_mem);

        /* Process all failed-step cases, and exit loop. */

        if sflag != IDA_SUCCESS {
            istate = IDAHandleFailure(IDA_mem, sflag);
            let tn = IDA_mem.borrow().ida_tn;
            IDA_mem.borrow_mut().ida_tretlast = tn;
            *tret = tn;
            let _ = IDAGetSolution(IDA_mem, tn, yret, ypret);
            break;
        }

        nstloc += 1;

        /* If tstop is set and was reached, reset IDA_mem->ida_tn = tstop */
        if IDA_mem.borrow().ida_tstopset {
            let mut m = IDA_mem.borrow_mut();
            let troundoff = HUNDRED * m.ida_uround * (SUNRabs(m.ida_tn) + SUNRabs(m.ida_hh));
            if SUNRabs(m.ida_tn - m.ida_tstop) <= troundoff {
                m.ida_tn = m.ida_tstop;
            }
        }

        /* After successful step, check for stop conditions; continue or break. */

        /* First check for root in the last step taken. */

        if IDA_mem.borrow().ida_nrtfn > 0 {
            let ier = IDARcheck3(IDA_mem, tout, itask);

            if ier == RTFOUND {
                /* A new root was found */
                let tlo = IDA_mem.borrow().ida_tlo;
                {
                    let mut m = IDA_mem.borrow_mut();
                    m.ida_irfnd = 1;
                    m.ida_tretlast = tlo;
                }
                istate = IDA_ROOT_RETURN;
                *tret = tlo;
                break;
            } else if ier == IDA_RTFUNC_FAIL {
                /* g failed */
                let tlo = IDA_mem.borrow().ida_tlo;
                IDAProcessError(
                    Some(IDA_mem),
                    IDA_RTFUNC_FAIL,
                    line!() as i32,
                    "IDASolve",
                    file!(),
                    &MSG_RTFUNC_FAILED(tlo),
                );
                istate = IDA_RTFUNC_FAIL;
                break;
            }

            /* If we are at the end of the first step and we still have
             * some event functions that are inactive, issue a warning
             * as this may indicate a user error in the implementation
             * of the root function. */

            if IDA_mem.borrow().ida_nst == 1 {
                let (inactive_roots, mxgnull) = {
                    let m = IDA_mem.borrow();
                    let mut inactive_roots = SUNFALSE;
                    for ir in 0..m.ida_nrtfn as usize {
                        if !m.ida_gactive[ir] {
                            inactive_roots = SUNTRUE;
                            break;
                        }
                    }
                    (inactive_roots, m.ida_mxgnull)
                };
                if (mxgnull > 0) && inactive_roots {
                    IDAProcessError(
                        Some(IDA_mem),
                        IDA_WARNING,
                        line!() as i32,
                        "IDASolve",
                        file!(),
                        MSG_INACTIVE_ROOTS,
                    );
                }
            }
        }

        /* Now check all other stop conditions. */

        istate = IDAStopTest2(IDA_mem, tout, tret, yret, ypret, itask);
        if istate != CONTINUE_STEPS {
            break;
        }
    } /* End of step loop */

    istate
}

/*
 * -----------------------------------------------------------------
 * Interpolated output and extraction functions
 * -----------------------------------------------------------------
 */

/*
 * IDAGetDky
 *
 * This routine evaluates the k-th derivative of y(t) as the value of
 * the k-th derivative of the interpolating polynomial at the independent
 * variable t, and stores the results in the vector dky.  It uses the current
 * independent variable value, tn, and the method order last used, kused.
 *
 * The return values are:
 *   IDA_SUCCESS       if t is legal
 *   IDA_BAD_T         if t is not within the interval of the last step taken
 *   IDA_BAD_DKY       if the dky vector is NULL
 *   IDA_BAD_K         if the requested k is not in the range [0,order used]
 *   IDA_VECTOROP_ERR  if the fused vector operation fails
 *
 */

pub fn IDAGetDky(ida_mem: &IDAMem, t: sunrealtype, k: i32, dky: &N_Vector) -> i32 {
    /* Check ida_mem */
    /* NULL-mem check: handled by type system */
    let IDA_mem = ida_mem;

    /* NULL dky check: handled by type system */

    if (k < 0) || (k > IDA_mem.borrow().ida_kused) {
        IDAProcessError(
            Some(IDA_mem),
            IDA_BAD_K,
            line!() as i32,
            "IDAGetDky",
            file!(),
            MSG_BAD_K,
        );
        return IDA_BAD_K;
    }

    /* Check t for legality.  Here tn - hused is t_{n-1}. */

    let (uround, tn, hh, hused, kused, psi) = {
        let m = IDA_mem.borrow();
        (
            m.ida_uround,
            m.ida_tn,
            m.ida_hh,
            m.ida_hused,
            m.ida_kused,
            m.ida_psi,
        )
    };

    let mut tfuzz = HUNDRED * uround * (SUNRabs(tn) + SUNRabs(hh));
    if hh < ZERO {
        tfuzz = -tfuzz;
    }
    let tp = tn - hused - tfuzz;
    if (t - tp) * hh < ZERO {
        IDAProcessError(
            Some(IDA_mem),
            IDA_BAD_T,
            line!() as i32,
            "IDAGetDky",
            file!(),
            &MSG_BAD_T(t, tn - hused, tn),
        );
        return IDA_BAD_T;
    }

    /* Initialize the c_j^(k) and c_k^(k-1) */
    let mut cjk = [ZERO; MXORDP1];
    let mut cjk_1 = [ZERO; MXORDP1];

    let delt = t - tn;
    let mut psij_1: sunrealtype;

    for i in 0..=k {
        /* The below recurrence is used to compute the k-th derivative of the solution:
           c_j^(k) = ( k * c_{j-1}^(k-1) + c_{j-1}^{k} (Delta+psi_{j-1}) ) / psi_j

           Translated in indexes notation:
           cjk[j] = ( k*cjk_1[j-1] + cjk[j-1]*(delt+psi[j-2]) ) / psi[j-1]

           For k=0, j=1: c_1 = c_0^(-1) + (delt+psi[-1]) / psi[0]

           In order to be able to deal with k=0 in the same way as for k>0, the
           following conventions were adopted:
             - c_0(t) = 1 , c_0^(-1)(t)=0
             - psij_1 stands for psi[-1]=0 when j=1
                             for psi[j-2]  when j>1
        */
        if i == 0 {
            cjk[i as usize] = ONE;
            psij_1 = ZERO;
        } else {
            /*                                                i       i-1          1
              c_i^(i) can be always updated since c_i^(i) = -----  --------  ... -----
                                                            psi_j  psi_{j-1}     psi_1
            */
            cjk[i as usize] = cjk[(i - 1) as usize] * (i as sunrealtype) / psi[(i - 1) as usize];
            psij_1 = psi[(i - 1) as usize];
        }

        /* update c_j^(i) */

        /*j does not need to go till kused */
        for j in (i + 1)..=(kused - k + i) {
            cjk[j as usize] = ((i as sunrealtype) * cjk_1[(j - 1) as usize]
                + cjk[(j - 1) as usize] * (delt + psij_1))
                / psi[(j - 1) as usize];
            psij_1 = psi[(j - 1) as usize];
        }

        /* save existing c_j^(i)'s */
        for j in (i + 1)..=(kused - k + i) {
            cjk_1[j as usize] = cjk[j as usize];
        }
    }

    /* Compute sum (c_j(t) * phi(t)) */

    /* Sum j=k to j<=IDA_mem->ida_kused */
    let Xvecs: Vec<N_Vector> = {
        let m = IDA_mem.borrow();
        ((k as usize)..=(kused as usize))
            .map(|j| m.ida_phi[j].clone().unwrap())
            .collect()
    };
    let retval = N_VLinearCombination(kused - k + 1, &cjk[(k as usize)..], &Xvecs, dky);
    if retval != IDA_SUCCESS {
        return IDA_VECTOROP_ERR;
    }

    IDA_SUCCESS
}

/*
 * IDAComputeY
 *
 * Computes y based on the current prediction and given correction.
 */
pub fn IDAComputeY(ida_mem: &IDAMem, ycor: &N_Vector, y: &N_Vector) -> i32 {
    /* NULL-mem check: handled by type system */
    let IDA_mem = ida_mem;

    let yypredict = IDA_mem.borrow().ida_yypredict.clone().unwrap();
    N_VLinearSum(ONE, &yypredict, ONE, ycor, y);

    IDA_SUCCESS
}

/*
 * IDAComputeYp
 *
 * Computes y' based on the current prediction and given correction.
 */
pub fn IDAComputeYp(ida_mem: &IDAMem, ycor: &N_Vector, yp: &N_Vector) -> i32 {
    /* NULL-mem check: handled by type system */
    let IDA_mem = ida_mem;

    let (yppredict, cj) = {
        let m = IDA_mem.borrow();
        (m.ida_yppredict.clone().unwrap(), m.ida_cj)
    };
    N_VLinearSum(ONE, &yppredict, cj, ycor, yp);

    IDA_SUCCESS
}

/*
 * -----------------------------------------------------------------
 * Deallocation function
 * -----------------------------------------------------------------
 */

/*
 * IDAFree
 *
 * This routine frees the problem memory allocated by IDAInit
 * Such memory includes all the vectors allocated by IDAAllocVectors,
 * and the memory lmem for the linear solver (deallocated by a call
 * to lfree).
 */

pub fn IDAFree(ida_mem: &mut Option<IDAMem>) {
    if ida_mem.is_none() {
        return;
    }

    let IDA_mem = ida_mem.as_ref().unwrap().clone();

    IDAFreeVectors(&IDA_mem);

    /* if IDA created the NLS object then free it */
    if IDA_mem.borrow().ownNLS {
        let nls = {
            let mut m = IDA_mem.borrow_mut();
            m.ownNLS = SUNFALSE;
            m.NLS.take()
        };
        let _ = SUNNonlinSolFree(nls);
    }

    let lfree = IDA_mem.borrow().ida_lfree;
    if let Some(lfree) = lfree {
        let _ = lfree(&IDA_mem);
    }

    if IDA_mem.borrow().ida_nrtfn > 0 {
        let mut m = IDA_mem.borrow_mut();
        m.ida_glo = Vec::new();
        m.ida_ghi = Vec::new();
        m.ida_grout = Vec::new();
        m.ida_iroots = Vec::new();
        m.ida_rootdir = Vec::new();
        m.ida_gactive = Vec::new();
    }

    /* C frees the mem struct wholesale; the Rust handle is dropped by the
    caller, so break the Rc cycle the default-efun edata token creates
    (ida_edata holds an IDAMem clone pointing back at this record) */
    IDA_mem.borrow_mut().ida_edata = None;

    *ida_mem = None;
}

/*
 * =================================================================
 * PRIVATE FUNCTIONS
 * =================================================================
 */

/*
 * IDACheckNvector
 *
 * This routine checks if all required vector operations are present.
 * If any of them is missing it returns SUNFALSE.
 */

fn IDACheckNvector(tmpl: &N_Vector) -> sunbooleantype {
    let ops = tmpl.ops.borrow();
    if ops.nvclone.is_none()
        || ops.nvdestroy.is_none()
        || ops.nvlinearsum.is_none()
        || ops.nvconst.is_none()
        || ops.nvprod.is_none()
        || ops.nvscale.is_none()
        || ops.nvabs.is_none()
        || ops.nvinv.is_none()
        || ops.nvaddconst.is_none()
        || ops.nvwrmsnorm.is_none()
        || ops.nvmin.is_none()
    {
        SUNFALSE
    } else {
        SUNTRUE
    }
}

/*
 * -----------------------------------------------------------------
 * Memory allocation/deallocation
 * -----------------------------------------------------------------
 */

/*
 * IDAAllocVectors
 *
 * This routine allocates the IDA vectors ewt, tempv1, tempv2, and
 * phi[0], ..., phi[maxord].
 * If all memory allocations are successful, IDAAllocVectors returns
 * SUNTRUE. Otherwise all allocated memory is freed and IDAAllocVectors
 * returns SUNFALSE.
 * This routine also sets the optional outputs lrw and liw, which are
 * (respectively) the lengths of the real and integer work spaces
 * allocated here.
 */

fn IDAAllocVectors(IDA_mem: &IDAMem, tmpl: &N_Vector) -> sunbooleantype {
    /* Allocate ewt, ee, delta, yypredict, yppredict, savres, tempv1, tempv2, tempv3 */

    let ewt = match N_VClone(tmpl) {
        Some(v) => v,
        None => return SUNFALSE,
    };

    let ee = match N_VClone(tmpl) {
        Some(v) => v,
        None => {
            N_VDestroy(ewt);
            return SUNFALSE;
        }
    };

    let delta = match N_VClone(tmpl) {
        Some(v) => v,
        None => {
            N_VDestroy(ewt);
            N_VDestroy(ee);
            return SUNFALSE;
        }
    };

    let yypredict = match N_VClone(tmpl) {
        Some(v) => v,
        None => {
            N_VDestroy(ewt);
            N_VDestroy(ee);
            N_VDestroy(delta);
            return SUNFALSE;
        }
    };

    let yppredict = match N_VClone(tmpl) {
        Some(v) => v,
        None => {
            N_VDestroy(ewt);
            N_VDestroy(ee);
            N_VDestroy(delta);
            N_VDestroy(yypredict);
            return SUNFALSE;
        }
    };

    let savres = match N_VClone(tmpl) {
        Some(v) => v,
        None => {
            N_VDestroy(ewt);
            N_VDestroy(ee);
            N_VDestroy(delta);
            N_VDestroy(yypredict);
            N_VDestroy(yppredict);
            return SUNFALSE;
        }
    };

    let tempv1 = match N_VClone(tmpl) {
        Some(v) => v,
        None => {
            N_VDestroy(ewt);
            N_VDestroy(ee);
            N_VDestroy(delta);
            N_VDestroy(yypredict);
            N_VDestroy(yppredict);
            N_VDestroy(savres);
            return SUNFALSE;
        }
    };

    let tempv2 = match N_VClone(tmpl) {
        Some(v) => v,
        None => {
            N_VDestroy(ewt);
            N_VDestroy(ee);
            N_VDestroy(delta);
            N_VDestroy(yypredict);
            N_VDestroy(yppredict);
            N_VDestroy(savres);
            N_VDestroy(tempv1);
            return SUNFALSE;
        }
    };

    let tempv3 = match N_VClone(tmpl) {
        Some(v) => v,
        None => {
            N_VDestroy(ewt);
            N_VDestroy(ee);
            N_VDestroy(delta);
            N_VDestroy(yypredict);
            N_VDestroy(yppredict);
            N_VDestroy(savres);
            N_VDestroy(tempv1);
            N_VDestroy(tempv2);
            return SUNFALSE;
        }
    };

    {
        let mut m = IDA_mem.borrow_mut();
        m.ida_ewt = Some(ewt);
        m.ida_ee = Some(ee);
        m.ida_delta = Some(delta);
        m.ida_yypredict = Some(yypredict);
        m.ida_yppredict = Some(yppredict);
        m.ida_savres = Some(savres);
        m.ida_tempv1 = Some(tempv1);
        m.ida_tempv2 = Some(tempv2);
        m.ida_tempv3 = Some(tempv3);
    }

    /* Allocate phi[0] ... phi[maxord].  Make sure phi[2] and phi[3] are
    allocated (for use as temporary vectors), regardless of maxord.       */

    let maxcol = SUNMAX(IDA_mem.borrow().ida_maxord, 3);
    for j in 0..=maxcol as usize {
        match N_VClone(tmpl) {
            Some(v) => {
                IDA_mem.borrow_mut().ida_phi[j] = Some(v);
            }
            None => {
                let mut m = IDA_mem.borrow_mut();
                if let Some(v) = m.ida_ewt.take() {
                    N_VDestroy(v);
                }
                if let Some(v) = m.ida_ee.take() {
                    N_VDestroy(v);
                }
                if let Some(v) = m.ida_delta.take() {
                    N_VDestroy(v);
                }
                if let Some(v) = m.ida_yypredict.take() {
                    N_VDestroy(v);
                }
                if let Some(v) = m.ida_yppredict.take() {
                    N_VDestroy(v);
                }
                if let Some(v) = m.ida_savres.take() {
                    N_VDestroy(v);
                }
                if let Some(v) = m.ida_tempv1.take() {
                    N_VDestroy(v);
                }
                if let Some(v) = m.ida_tempv2.take() {
                    N_VDestroy(v);
                }
                if let Some(v) = m.ida_tempv3.take() {
                    N_VDestroy(v);
                }
                for i in 0..j {
                    if let Some(v) = m.ida_phi[i].take() {
                        N_VDestroy(v);
                    }
                }
                return SUNFALSE;
            }
        }
    }

    {
        let mut m = IDA_mem.borrow_mut();

        /* Update solver workspace lengths  */
        m.ida_lrw += ((maxcol + 10) as i64) * m.ida_lrw1;
        m.ida_liw += ((maxcol + 10) as i64) * m.ida_liw1;

        /* Store the value of maxord used here */
        m.ida_maxord_alloc = m.ida_maxord;
    }

    SUNTRUE
}

/*
 * IDAfreeVectors
 *
 * This routine frees the IDA vectors allocated for IDA.
 */

fn IDAFreeVectors(IDA_mem: &IDAMem) {
    let mut m = IDA_mem.borrow_mut();

    if let Some(v) = m.ida_ewt.take() {
        N_VDestroy(v);
    }
    if let Some(v) = m.ida_ee.take() {
        N_VDestroy(v);
    }
    if let Some(v) = m.ida_delta.take() {
        N_VDestroy(v);
    }
    if let Some(v) = m.ida_yypredict.take() {
        N_VDestroy(v);
    }
    if let Some(v) = m.ida_yppredict.take() {
        N_VDestroy(v);
    }
    if let Some(v) = m.ida_savres.take() {
        N_VDestroy(v);
    }
    if let Some(v) = m.ida_tempv1.take() {
        N_VDestroy(v);
    }
    if let Some(v) = m.ida_tempv2.take() {
        N_VDestroy(v);
    }
    if let Some(v) = m.ida_tempv3.take() {
        N_VDestroy(v);
    }
    let maxcol = SUNMAX(m.ida_maxord_alloc, 3);
    for j in 0..=maxcol as usize {
        if let Some(v) = m.ida_phi[j].take() {
            N_VDestroy(v);
        }
    }

    m.ida_lrw -= ((maxcol + 10) as i64) * m.ida_lrw1;
    m.ida_liw -= ((maxcol + 10) as i64) * m.ida_liw1;

    if m.ida_VatolMallocDone {
        if let Some(v) = m.ida_Vatol.take() {
            N_VDestroy(v);
        }
        m.ida_lrw -= m.ida_lrw1;
        m.ida_liw -= m.ida_liw1;
    }

    if m.ida_constraints.is_some() {
        if let Some(v) = m.ida_constraints.take() {
            N_VDestroy(v);
        }
        m.ida_lrw -= m.ida_lrw1;
        m.ida_liw -= m.ida_liw1;
    }

    if m.ida_idMallocDone {
        if let Some(v) = m.ida_id.take() {
            N_VDestroy(v);
        }
        m.ida_lrw -= m.ida_lrw1;
        m.ida_liw -= m.ida_liw1;
    }
}

/*
 * -----------------------------------------------------------------
 * Initial setup
 * -----------------------------------------------------------------
 */

/*
 * IDAInitialSetup
 *
 * This routine is called by IDASolve once at the first step.
 * It performs all checks on optional inputs and inputs to
 * IDAInit/IDAReInit that could not be done before.
 *
 * If no error is encountered, IDAInitialSetup returns IDA_SUCCESS.
 * Otherwise, it returns an error flag and reported to the error
 * handler function.
 */

pub fn IDAInitialSetup(IDA_mem: &IDAMem) -> i32 {
    /* Test for more vector operations, depending on options */
    if IDA_mem.borrow().ida_suppressalg {
        let phi0 = IDA_mem.borrow().ida_phi[0].clone().unwrap();
        if phi0.ops.borrow().nvwrmsnormmask.is_none() {
            IDAProcessError(
                Some(IDA_mem),
                IDA_ILL_INPUT,
                line!() as i32,
                "IDAInitialSetup",
                file!(),
                MSG_BAD_NVECTOR,
            );
            return IDA_ILL_INPUT;
        }
    }

    /* Test id vector for legality */
    if IDA_mem.borrow().ida_suppressalg && IDA_mem.borrow().ida_id.is_none() {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDAInitialSetup",
            file!(),
            MSG_MISSING_ID,
        );
        return IDA_ILL_INPUT;
    }

    /* Did the user specify tolerances? */
    if IDA_mem.borrow().ida_itol == IDA_NN {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDAInitialSetup",
            file!(),
            MSG_NO_TOLS,
        );
        return IDA_ILL_INPUT;
    }

    /* Set data for efun */
    if IDA_mem.borrow().ida_user_efun {
        /* C: ida_edata = ida_user_data (pointer alias); efun call sites pass
        ida_user_data directly instead (box aliasing impossible) */
        IDA_mem.borrow_mut().ida_edata = None;
    } else {
        let token: Box<dyn Any> = Box::new(IDA_mem.clone());
        IDA_mem.borrow_mut().ida_edata = Some(token);
    }

    /* Initial error weight vector */
    let (phi0, ewt) = {
        let m = IDA_mem.borrow();
        (m.ida_phi[0].clone().unwrap(), m.ida_ewt.clone().unwrap())
    };
    let ier = ida_call_efun(IDA_mem, &phi0, &ewt);
    if ier != 0 {
        if IDA_mem.borrow().ida_itol == IDA_WF {
            IDAProcessError(
                Some(IDA_mem),
                IDA_ILL_INPUT,
                line!() as i32,
                "IDAInitialSetup",
                file!(),
                MSG_FAIL_EWT,
            );
        } else {
            IDAProcessError(
                Some(IDA_mem),
                IDA_ILL_INPUT,
                line!() as i32,
                "IDAInitialSetup",
                file!(),
                MSG_BAD_EWT,
            );
        }
        return IDA_ILL_INPUT;
    }

    /* Check to see if y0 satisfies constraints. */
    let constraints = IDA_mem.borrow().ida_constraints.clone();
    if let Some(constraints) = constraints {
        let (phi0, tempv2) = {
            let m = IDA_mem.borrow();
            (m.ida_phi[0].clone().unwrap(), m.ida_tempv2.clone().unwrap())
        };
        let conOK = N_VConstrMask(&constraints, &phi0, &tempv2);
        if !conOK {
            IDAProcessError(
                Some(IDA_mem),
                IDA_ILL_INPUT,
                line!() as i32,
                "IDAInitialSetup",
                file!(),
                MSG_Y0_FAIL_CONSTR,
            );
            return IDA_ILL_INPUT;
        }
    }

    /* Call linit function if it exists. */
    let linit = IDA_mem.borrow().ida_linit;
    if let Some(linit) = linit {
        let ier = linit(IDA_mem);
        if ier != 0 {
            IDAProcessError(
                Some(IDA_mem),
                IDA_LINIT_FAIL,
                line!() as i32,
                "IDAInitialSetup",
                file!(),
                MSG_LINIT_FAIL,
            );
            return IDA_LINIT_FAIL;
        }
    }

    /* Initialize the nonlinear solver (must occur after linear solver is initialize) so
     * that lsetup and lsolve pointers have been set */
    let ier = crate::ida_nls::idaNlsInit(IDA_mem);
    if ier != IDA_SUCCESS {
        IDAProcessError(
            Some(IDA_mem),
            IDA_NLS_INIT_FAIL,
            line!() as i32,
            "IDAInitialSetup",
            file!(),
            MSG_NLS_INIT_FAIL,
        );
        return IDA_NLS_INIT_FAIL;
    }

    IDA_SUCCESS
}

/*
 * IDAEwtSet
 *
 * This routine is responsible for loading the error weight vector
 * ewt, according to itol, as follows:
 * (1) ewt[i] = 1 / (rtol * SUNRabs(ycur[i]) + atol), i=0,...,Neq-1
 *     if itol = IDA_SS
 * (2) ewt[i] = 1 / (rtol * SUNRabs(ycur[i]) + atol[i]), i=0,...,Neq-1
 *     if itol = IDA_SV
 *
 *  IDAEwtSet returns 0 if ewt is successfully set as above to a
 *  positive vector and -1 otherwise. In the latter case, ewt is
 *  considered undefined.
 *
 * All the real work is done in the routines IDAEwtSetSS, IDAEwtSetSV.
 */

pub fn IDAEwtSet(ycur: &N_Vector, weight: &N_Vector, data: &mut Option<Box<dyn Any>>) -> i32 {
    /* data points to IDA_mem here (a boxed IDAMem handle clone; C's cast
    of a NULL/foreign pointer is UB -> deterministic panic) */
    let IDA_mem = data
        .as_mut()
        .and_then(|b| b.downcast_ref::<IDAMem>())
        .cloned()
        .expect("IDAEwtSet data holds IDAMem");

    let itol = IDA_mem.borrow().ida_itol;
    let flag: i32 = match itol {
        IDA_SS => IDAEwtSetSS(&IDA_mem, ycur, weight),
        IDA_SV => IDAEwtSetSV(&IDA_mem, ycur, weight),
        _ => 0,
    };

    flag
}

/*
 * IDAEwtSetSS
 *
 * This routine sets ewt as described above in the case itol=IDA_SS.
 * If the absolute tolerance is zero, it tests for non-positive components
 * before inverting. IDAEwtSetSS returns 0 if ewt is successfully set to a
 * positive vector and -1 otherwise. In the latter case, ewt is considered
 * undefined.
 */

fn IDAEwtSetSS(IDA_mem: &IDAMem, ycur: &N_Vector, weight: &N_Vector) -> i32 {
    let (tempv1, rtol, Satol, atolmin0) = {
        let m = IDA_mem.borrow();
        (
            m.ida_tempv1.clone().unwrap(),
            m.ida_rtol,
            m.ida_Satol,
            m.ida_atolmin0,
        )
    };
    N_VAbs(ycur, &tempv1);
    N_VScale(rtol, &tempv1, &tempv1);
    N_VAddConst(&tempv1, Satol, &tempv1);
    if atolmin0 {
        if N_VMin(&tempv1) <= ZERO {
            return -1;
        }
    }
    N_VInv(&tempv1, weight);

    0
}

/*
 * IDAEwtSetSV
 *
 * This routine sets ewt as described above in the case itol=IDA_SV.
 * If the absolute tolerance is zero, it tests for non-positive components
 * before inverting. IDAEwtSetSV returns 0 if ewt is successfully set to a
 * positive vector and -1 otherwise. In the latter case, ewt is considered
 * undefined.
 */

fn IDAEwtSetSV(IDA_mem: &IDAMem, ycur: &N_Vector, weight: &N_Vector) -> i32 {
    let (tempv1, rtol, Vatol, atolmin0) = {
        let m = IDA_mem.borrow();
        (
            m.ida_tempv1.clone().unwrap(),
            m.ida_rtol,
            m.ida_Vatol.clone().unwrap(),
            m.ida_atolmin0,
        )
    };
    N_VAbs(ycur, &tempv1);
    N_VLinearSum(rtol, &tempv1, ONE, &Vatol, &tempv1);
    if atolmin0 {
        if N_VMin(&tempv1) <= ZERO {
            return -1;
        }
    }
    N_VInv(&tempv1, weight);

    0
}

/*
 * -----------------------------------------------------------------
 * Stopping tests
 * -----------------------------------------------------------------
 */

/*
 * IDAStopTest1
 *
 * This routine tests for stop conditions before taking a step.
 * The tests depend on the value of itask.
 * The variable tretlast is the previously returned value of tret.
 *
 * The return values are:
 * CONTINUE_STEPS       if no stop conditions were found
 * IDA_SUCCESS          for a normal return to the user
 * IDA_TSTOP_RETURN     for a tstop-reached return to the user
 * IDA_ILL_INPUT        for an illegal-input return to the user
 *
 * In the tstop cases, this routine may adjust the stepsize hh to cause
 * the next step to reach tstop exactly.
 */

fn IDAStopTest1(
    IDA_mem: &IDAMem,
    tout: sunrealtype,
    tret: &mut sunrealtype,
    yret: &N_Vector,
    ypret: &N_Vector,
    itask: i32,
) -> i32 {
    if IDA_mem.borrow().ida_tstopset {
        /* Test for tn past tstop */
        {
            let m = IDA_mem.borrow();
            if (m.ida_tn - m.ida_tstop) * m.ida_hh > ZERO {
                let (tstop, tn) = (m.ida_tstop, m.ida_tn);
                drop(m);
                IDAProcessError(
                    Some(IDA_mem),
                    IDA_ILL_INPUT,
                    line!() as i32,
                    "IDAStopTest1",
                    file!(),
                    &MSG_BAD_TSTOP(tstop, tn),
                );
                return IDA_ILL_INPUT;
            }
        }

        let (tn, tstop, hh, uround) = {
            let m = IDA_mem.borrow();
            (m.ida_tn, m.ida_tstop, m.ida_hh, m.ida_uround)
        };
        let troundoff = HUNDRED * uround * (SUNRabs(tn) + SUNRabs(hh));

        /* Test for tn at tstop */
        if SUNRabs(tn - tstop) <= troundoff {
            /* Ensure tout >= tstop, otherwise check for tout return below */
            if (tout - tstop) * hh >= ZERO || SUNRabs(tout - tstop) <= troundoff {
                let ier = IDAGetSolution(IDA_mem, tstop, yret, ypret);
                if ier != IDA_SUCCESS {
                    IDAProcessError(
                        Some(IDA_mem),
                        IDA_ILL_INPUT,
                        line!() as i32,
                        "IDAStopTest1",
                        file!(),
                        &MSG_BAD_TSTOP(tstop, tn),
                    );
                    return IDA_ILL_INPUT;
                }
                {
                    let mut m = IDA_mem.borrow_mut();
                    m.ida_tretlast = tstop;
                    m.ida_tstopset = SUNFALSE;
                }
                *tret = tstop;
                return IDA_TSTOP_RETURN;
            }
        }
        /* Test for tn approaching tstop */
        else if (tn + hh - tstop) * hh > ZERO {
            IDA_mem.borrow_mut().ida_hh = (tstop - tn) * (ONE - FOUR * uround);
        }
    }

    match itask {
        IDA_NORMAL => {
            /* Test for tout = tretlast, and for tn past tout. */
            if tout == IDA_mem.borrow().ida_tretlast {
                IDA_mem.borrow_mut().ida_tretlast = tout;
                *tret = tout;
                return IDA_SUCCESS;
            }
            let past_tout = {
                let m = IDA_mem.borrow();
                (m.ida_tn - tout) * m.ida_hh >= ZERO
            };
            if past_tout {
                let ier = IDAGetSolution(IDA_mem, tout, yret, ypret);
                if ier != IDA_SUCCESS {
                    IDAProcessError(
                        Some(IDA_mem),
                        IDA_ILL_INPUT,
                        line!() as i32,
                        "IDAStopTest1",
                        file!(),
                        &MSG_BAD_TOUT(tout),
                    );
                    return IDA_ILL_INPUT;
                }
                IDA_mem.borrow_mut().ida_tretlast = tout;
                *tret = tout;
                return IDA_SUCCESS;
            }

            CONTINUE_STEPS
        }

        IDA_ONE_STEP => {
            /* Test for tn past tretlast. */
            let past_tretlast = {
                let m = IDA_mem.borrow();
                (m.ida_tn - m.ida_tretlast) * m.ida_hh > ZERO
            };
            if past_tretlast {
                let tn = IDA_mem.borrow().ida_tn;
                let _ = IDAGetSolution(IDA_mem, tn, yret, ypret);
                IDA_mem.borrow_mut().ida_tretlast = tn;
                *tret = tn;
                return IDA_SUCCESS;
            }

            CONTINUE_STEPS
        }

        _ => IDA_ILL_INPUT, /* This return should never happen. */
    }
}

/*
 * IDAStopTest2
 *
 * This routine tests for stop conditions after taking a step.
 * The tests depend on the value of itask.
 *
 * The return values are:
 *  CONTINUE_STEPS     if no stop conditions were found
 *  IDA_SUCCESS        for a normal return to the user
 *  IDA_TSTOP_RETURN   for a tstop-reached return to the user
 *  IDA_ILL_INPUT      for an illegal-input return to the user
 *
 * In the two cases with tstop, this routine may reset the stepsize hh
 * to cause the next step to reach tstop exactly.
 *
 * In the two cases with ONE_STEP mode, no interpolation to tn is needed
 * because yret and ypret already contain the current y and y' values.
 *
 * Note: No test is made for an error return from IDAGetSolution here,
 * because the same test was made prior to the step.
 */

fn IDAStopTest2(
    IDA_mem: &IDAMem,
    tout: sunrealtype,
    tret: &mut sunrealtype,
    yret: &N_Vector,
    ypret: &N_Vector,
    itask: i32,
) -> i32 {
    /* int ier; */

    if IDA_mem.borrow().ida_tstopset {
        let (tn, tstop, hh, uround) = {
            let m = IDA_mem.borrow();
            (m.ida_tn, m.ida_tstop, m.ida_hh, m.ida_uround)
        };
        let troundoff = HUNDRED * uround * (SUNRabs(tn) + SUNRabs(hh));

        /* Test for tn at tstop */
        if SUNRabs(tn - tstop) <= troundoff {
            /* Ensure tout >= tstop, otherwise check for tout return below */
            if (tout - tstop) * hh >= ZERO || SUNRabs(tout - tstop) <= troundoff {
                /* ier = */
                let _ = IDAGetSolution(IDA_mem, tstop, yret, ypret);
                {
                    let mut m = IDA_mem.borrow_mut();
                    m.ida_tretlast = tstop;
                    m.ida_tstopset = SUNFALSE;
                }
                *tret = tstop;
                return IDA_TSTOP_RETURN;
            }
        }
        /* Test for tn approaching tstop */
        else if (tn + hh - tstop) * hh > ZERO {
            IDA_mem.borrow_mut().ida_hh = (tstop - tn) * (ONE - FOUR * uround);
        }
    }

    match itask {
        IDA_NORMAL => {
            /* Test for tn past tout. */
            let past_tout = {
                let m = IDA_mem.borrow();
                (m.ida_tn - tout) * m.ida_hh >= ZERO
            };
            if past_tout {
                /* ier = */
                let _ = IDAGetSolution(IDA_mem, tout, yret, ypret);
                IDA_mem.borrow_mut().ida_tretlast = tout;
                *tret = tout;
                return IDA_SUCCESS;
            }

            CONTINUE_STEPS
        }

        IDA_ONE_STEP => {
            let tn = IDA_mem.borrow().ida_tn;
            IDA_mem.borrow_mut().ida_tretlast = tn;
            *tret = tn;
            IDA_SUCCESS
        }

        _ => IDA_ILL_INPUT, /* This return should never happen. */
    }
}

/* =================================================================
 * ida.c PART B -- fragment of the `ida` module.
 *
 * Covers every function of `src/ida/ida.c` whose definition begins at
 * line 2300 or later: IDAHandleFailure, IDAStep, IDASetCoeffs, IDANls,
 * IDACheckConstraints, IDAPredict, IDATestError, IDARestore,
 * IDAHandleNFlag, IDAReset, IDACompleteStep, IDAGetSolution,
 * IDAWrmsNorm, IDARcheck1, IDARcheck2, IDARcheck3, IDARootfind.
 * (`IDAProcessError`, defined at ida.c:4040, is relocated to
 * `ida_impl.rs` per the frozen contract and is NOT redefined here.)
 *
 * Fragment protocol: no `use` items and no module-scope consts -- the
 * concatenation target `ida.rs` supplies them; anything exotic is
 * spelled with a fully-qualified `sundials_core::...` path.
 *
 * Reference build: SUNDIALS_LOGGING_LEVEL = 2 (SUNLogInfo/SUNLogInfoIf/
 * SUNLogDebug call sites omitted at translation time), profiling off,
 * error checks off, monitoring on, serial branches only.
 * =================================================================*/

/*
 * -----------------------------------------------------------------
 * Error handler
 * -----------------------------------------------------------------
 */

/*
 * IDAHandleFailure
 *
 * This routine prints error messages for all cases of failure by
 * IDAStep.  It returns to IDASolve the value that it is to return to
 * the user.
 */

fn IDAHandleFailure(IDA_mem: &IDAMem, sflag: i32) -> i32 {
    /* C re-reads ida_tn / ida_hh at each IDAProcessError call site; nothing
    between the switch entry and the call mutates them. */
    let (tn, hh) = {
        let m = IDA_mem.borrow();
        (m.ida_tn, m.ida_hh)
    };

    /* Depending on sflag, print error message and return error flag */
    match sflag {
        IDA_ERR_FAIL => {
            IDAProcessError(
                Some(IDA_mem),
                IDA_ERR_FAIL,
                line!() as i32,
                "IDAHandleFailure",
                file!(),
                &MSG_ERR_FAILS(tn, hh),
            );
            return IDA_ERR_FAIL;
        }

        IDA_CONV_FAIL => {
            IDAProcessError(
                Some(IDA_mem),
                IDA_CONV_FAIL,
                line!() as i32,
                "IDAHandleFailure",
                file!(),
                &MSG_CONV_FAILS(tn, hh),
            );
            return IDA_CONV_FAIL;
        }

        IDA_LSETUP_FAIL => {
            IDAProcessError(
                Some(IDA_mem),
                IDA_LSETUP_FAIL,
                line!() as i32,
                "IDAHandleFailure",
                file!(),
                &MSG_SETUP_FAILED(tn),
            );
            return IDA_LSETUP_FAIL;
        }

        IDA_LSOLVE_FAIL => {
            IDAProcessError(
                Some(IDA_mem),
                IDA_LSOLVE_FAIL,
                line!() as i32,
                "IDAHandleFailure",
                file!(),
                &MSG_SOLVE_FAILED(tn),
            );
            return IDA_LSOLVE_FAIL;
        }

        IDA_REP_RES_ERR => {
            IDAProcessError(
                Some(IDA_mem),
                IDA_REP_RES_ERR,
                line!() as i32,
                "IDAHandleFailure",
                file!(),
                &MSG_REP_RES_ERR(tn),
            );
            return IDA_REP_RES_ERR;
        }

        IDA_RES_FAIL => {
            IDAProcessError(
                Some(IDA_mem),
                IDA_RES_FAIL,
                line!() as i32,
                "IDAHandleFailure",
                file!(),
                &MSG_RES_NONRECOV(tn),
            );
            return IDA_RES_FAIL;
        }

        IDA_CONSTR_FAIL => {
            IDAProcessError(
                Some(IDA_mem),
                IDA_CONSTR_FAIL,
                line!() as i32,
                "IDAHandleFailure",
                file!(),
                &MSG_FAILED_CONSTR(tn),
            );
            return IDA_CONSTR_FAIL;
        }

        IDA_MEM_NULL => {
            IDAProcessError(
                None,
                IDA_MEM_NULL,
                line!() as i32,
                "IDAHandleFailure",
                file!(),
                MSG_NO_MEM,
            );
            return IDA_MEM_NULL;
        }

        sundials_core::sundials_errors::SUN_ERR_ARG_CORRUPT => {
            IDAProcessError(
                Some(IDA_mem),
                IDA_MEM_NULL,
                line!() as i32,
                "IDAHandleFailure",
                file!(),
                &MSG_NLS_INPUT_NULL(tn),
            );
            return IDA_MEM_NULL;
        }

        IDA_NLS_SETUP_FAIL => {
            IDAProcessError(
                Some(IDA_mem),
                IDA_NLS_SETUP_FAIL,
                line!() as i32,
                "IDAHandleFailure",
                file!(),
                &MSG_NLS_SETUP_FAILED(tn),
            );
            return IDA_NLS_SETUP_FAIL;
        }
        IDA_NLS_FAIL => {
            IDAProcessError(
                Some(IDA_mem),
                IDA_NLS_FAIL,
                line!() as i32,
                "IDAHandleFailure",
                file!(),
                &MSG_NLS_FAIL(tn),
            );
            return IDA_NLS_FAIL;
        }
        _ => {}
    }

    /* This return should never happen */
    IDAProcessError(
        Some(IDA_mem),
        IDA_UNRECOGNIZED_ERROR,
        line!() as i32,
        "IDAHandleFailure",
        file!(),
        "IDA encountered an unrecognized error. Please report this to the \
         Sundials developers at sundials-users@llnl.gov",
    );
    IDA_UNRECOGNIZED_ERROR
}

/*
 * -----------------------------------------------------------------
 * Main IDAStep function
 * -----------------------------------------------------------------
 */

/*
 * IDAStep
 *
 * This routine performs one internal IDA step, from tn to tn + hh.
 * It calls other routines to do all the work.
 *
 * It solves a system of differential/algebraic equations of the form
 *       F(t,y,y') = 0, for one step. In IDA, tt is used for t,
 * yy is used for y, and yp is used for y'. The function F is supplied as 'res'
 * by the user.
 *
 * The methods used are modified divided difference, fixed leading
 * coefficient forms of backward differentiation formulas.
 * The code adjusts the stepsize and order to control the local error per step.
 *
 * The main operations done here are as follows:
 *  * initialize various quantities;
 *  * setting of multistep method coefficients;
 *  * solution of the nonlinear system for yy at t = tn + hh;
 *  * deciding on order reduction and testing the local error;
 *  * attempting to recover from failure in nonlinear solver or error test;
 *  * resetting stepsize and order for the next step.
 *  * updating phi and other state data if successful;
 *
 * On a failure in the nonlinear system solution or error test, the
 * step may be reattempted, depending on the nature of the failure.
 *
 *       Return values are:
 *       IDA_SUCCESS   IDA_RES_FAIL      LSETUP_ERROR_NONRECVR
 *                     IDA_LSOLVE_FAIL   IDA_ERR_FAIL
 *                     IDA_CONSTR_FAIL   IDA_CONV_FAIL
 *                     IDA_REP_RES_ERR
 */

fn IDAStep(IDA_mem: &IDAMem) -> i32 {
    /* C declares `ck` uninitialized; IDASetCoeffs writes it through the
    out-parameter before every read. */
    let mut ck: sunrealtype = ZERO;

    let saved_t = IDA_mem.borrow().ida_tn;

    /* Initialize failure counters for this step attempt */

    let mut ncf: i32 = 0; /* corrector failures  */
    let mut nef: i32 = 0; /* error test failures */
    let mut step_constraint_fails: i32 = 0;

    if IDA_mem.borrow().ida_nst == 0 {
        let mut m = IDA_mem.borrow_mut();
        m.ida_kk = 1;
        m.ida_kused = 0;
        m.ida_hused = ZERO;
        m.ida_psi[0] = m.ida_hh;
        m.ida_cj = ONE / m.ida_hh;
        m.ida_phase = 0;
        m.ida_ns = 0;
    }

    /* To prevent 'unintialized variable' warnings */
    let mut err_k: sunrealtype = ZERO;
    let mut err_km1: sunrealtype = ZERO;

    /* Looping point for attempts to take a step */

    loop {
        /*-----------------------
        Set method coefficients
        -----------------------*/

        IDASetCoeffs(IDA_mem, &mut ck);

        /* (C sets kflag = IDA_SUCCESS here; the only read of kflag follows the
        IDAHandleNFlag assignment below, so the dead store is omitted.) */

        /*----------------------------------------------------
        If tn is past tstop (by roundoff), reset it to tstop.
        -----------------------------------------------------*/

        {
            let mut m = IDA_mem.borrow_mut();
            m.ida_tn = m.ida_tn + m.ida_hh;
            if m.ida_tstopset {
                if (m.ida_tn - m.ida_tstop) * m.ida_hh > ZERO {
                    m.ida_tn = m.ida_tstop;
                }
            }
        }

        /*-----------------------
        Advance state variables
        -----------------------*/

        /* Compute predicted values for yy and yp */
        IDAPredict(IDA_mem);

        /* Nonlinear system solution */
        let mut nflag = IDANls(IDA_mem);

        /* Nonlinear solve was successful */
        if nflag == IDA_SUCCESS {
            /* Check and enforce inequality constraints */
            if IDA_mem.borrow().ida_constraints.is_some() {
                nflag = IDACheckConstraints(IDA_mem, saved_t, &mut step_constraint_fails);

                /* Constraint check failed, predict again */
                if nflag == PREDICT_AGAIN {
                    continue;
                }

                /* Exit on nonrecoverable failure */
                if nflag != IDA_SUCCESS {
                    return nflag;
                }
            }

            /* Perform error test */
            nflag = IDATestError(IDA_mem, ck, &mut err_k, &mut err_km1);
        }

        /* Test for convergence or error test failures */
        if nflag != IDA_SUCCESS {
            /* restore and decide what to do */
            IDARestore(IDA_mem, saved_t);
            let kflag = IDAHandleNFlag(IDA_mem, nflag, err_k, err_km1, &mut ncf, &mut nef);

            /* exit on nonrecoverable failure */
            if kflag != PREDICT_AGAIN {
                return kflag;
            }

            /* recoverable error; predict again */
            if IDA_mem.borrow().ida_nst == 0 {
                IDAReset(IDA_mem);
            }
            continue;
        }

        /* kflag == IDA_SUCCESS */
        break;
    } /* end loop */

    /* Nonlinear system solve and error test were both successful;
    update data, and consider change of step and/or order */

    IDACompleteStep(IDA_mem, err_k, err_km1);

    /*
    Rescale ee vector to be the estimated local error
    Notes:
      (1) altering the value of ee is permissible since
          it will be overwritten by
          IDASolve()->IDAStep()->IDANls()
          before it is needed again
      (2) the value of ee is only valid if IDAHandleNFlag()
          returns either PREDICT_AGAIN or IDA_SUCCESS
    */

    let ee = IDA_mem.borrow().ida_ee.clone().unwrap();
    N_VScale(ck, &ee, &ee);

    IDA_SUCCESS
}

/*
 * IDASetCoeffs
 *
 *  This routine computes the coefficients relevant to the current step.
 *  The counter ns counts the number of consecutive steps taken at
 *  constant stepsize h and order k, up to a maximum of k + 2.
 *  Then the first ns components of beta will be one, and on a step
 *  with ns = k + 2, the coefficients alpha, etc. need not be reset here.
 *  Also, IDACompleteStep prohibits an order increase until ns = k + 2.
 */

fn IDASetCoeffs(IDA_mem: &IDAMem, ck: &mut sunrealtype) {
    let (ns, kk) = {
        let mut m = IDA_mem.borrow_mut();

        /* Set coefficients for the current stepsize h */

        if (m.ida_hh != m.ida_hused) || (m.ida_kk != m.ida_kused) {
            m.ida_ns = 0;
        }
        m.ida_ns = SUNMIN(m.ida_ns + 1, m.ida_kused + 2);

        let kk = m.ida_kk as usize;

        if m.ida_kk + 1 >= m.ida_ns {
            m.ida_beta[0] = ONE;
            m.ida_alpha[0] = ONE;
            let mut temp1 = m.ida_hh;
            m.ida_gamma[0] = ZERO;
            m.ida_sigma[0] = ONE;
            for i in 1..=kk {
                let temp2 = m.ida_psi[i - 1];
                m.ida_psi[i - 1] = temp1;
                m.ida_beta[i] = m.ida_beta[i - 1] * m.ida_psi[i - 1] / temp2;
                temp1 = temp2 + m.ida_hh;
                m.ida_alpha[i] = m.ida_hh / temp1;
                m.ida_sigma[i] = i as sunrealtype * m.ida_sigma[i - 1] * m.ida_alpha[i];
                m.ida_gamma[i] = m.ida_gamma[i - 1] + m.ida_alpha[i - 1] / m.ida_hh;
            }
            m.ida_psi[kk] = temp1;
        }

        /* compute alphas, alpha0 */
        let mut alphas = ZERO;
        let mut alpha0 = ZERO;
        for i in 0..kk {
            alphas = alphas - ONE / ((i + 1) as sunrealtype);
            alpha0 = alpha0 - m.ida_alpha[i];
        }

        /* compute leading coefficient cj  */
        m.ida_cjlast = m.ida_cj;
        m.ida_cj = -alphas / m.ida_hh;

        /* compute variable stepsize error coefficient ck */

        *ck = SUNRabs(m.ida_alpha[kk] + alphas - alpha0);
        *ck = SUNMAX(*ck, m.ida_alpha[kk]);

        (m.ida_ns, m.ida_kk)
    };

    /* change phi to phi-star  */

    /* Scale i=IDA_mem->ida_ns to i<=IDA_mem->ida_kk */
    if ns <= kk {
        let ns = ns as usize;
        let kk = kk as usize;
        let (cvals, phivecs) = {
            let m = IDA_mem.borrow();
            let cvals: Vec<sunrealtype> = m.ida_beta[ns..=kk].to_vec();
            let phivecs: Vec<N_Vector> = (ns..=kk).map(|j| m.ida_phi[j].clone().unwrap()).collect();
            (cvals, phivecs)
        };
        /* C passes the same `ida_phi + ns` pointer for X and Z; the port passes
        the same slice so the array-pointer equality test in the vector op
        selects the identical in-place branch. */
        let _ = N_VScaleVectorArray((kk - ns + 1) as i32, &cvals, &phivecs, &phivecs);
    }
}

/*
 * -----------------------------------------------------------------
 * Nonlinear solver functions
 * -----------------------------------------------------------------
 */

/*
 * IDANls
 *
 * This routine attempts to solve the nonlinear system using the linear
 * solver specified. NOTE: this routine uses N_Vector ee as the scratch
 * vector tempv3 passed to lsetup.
 *
 *  Possible return values:
 *
 *  IDA_SUCCESS
 *
 *  IDA_RES_RECVR       IDA_RES_FAIL
 *  IDA_LSETUP_RECVR    IDA_LSETUP_FAIL
 *  IDA_LSOLVE_RECVR    IDA_LSOLVE_FAIL
 *
 *  SUN_NLS_CONV_RECVR
 *  IDA_MEM_NULL
 */

fn IDANls(IDA_mem: &IDAMem) -> i32 {
    let mut nni_inc: i64 = 0;
    let mut nnf_inc: i64 = 0;

    let mut callLSetup: sunbooleantype = SUNFALSE;

    {
        let mut m = IDA_mem.borrow_mut();

        /* Initialize if the first time called */

        if m.ida_nst == 0 {
            m.ida_cjold = m.ida_cj;
            m.ida_ss = TWENTY;
            if m.ida_lsetup.is_some() {
                callLSetup = SUNTRUE;
            }
        }

        /* Decide if lsetup is to be called */

        if m.ida_lsetup.is_some() {
            m.ida_cjratio = m.ida_cj / m.ida_cjold;
            let temp1 = (ONE - m.ida_dcj) / (ONE + m.ida_dcj);
            let temp2 = ONE / temp1;
            if m.ida_cjratio < temp1 || m.ida_cjratio > temp2 {
                callLSetup = SUNTRUE;
            }
            if m.ida_cj != m.ida_cjlast {
                m.ida_ss = HUNDRED;
            }
        }
    }

    /* initial guess for the correction to the predictor */
    let ee = IDA_mem.borrow().ida_ee.clone().unwrap();
    N_VConst(ZERO, &ee);

    /* The C `void* IDA_mem` handed to the nonlinear solver maps to a boxed
    handle clone (the same token shape ida_nls.rs downcasts). */
    let NLS = IDA_mem.borrow().NLS.clone().unwrap();
    let mut nls_mem: Option<Box<dyn std::any::Any>> = Some(Box::new(IDA_mem.clone()));

    /* call nonlinear solver setup if it exists */
    if NLS.ops.borrow().setup.is_some() {
        let retval = SUNNonlinSolSetup(&NLS, &ee, &mut nls_mem);
        if retval < 0 {
            return IDA_NLS_SETUP_FAIL;
        }
        if retval > 0 {
            return IDA_NLS_SETUP_RECVR;
        }
    }

    /* solve the nonlinear system */
    let (yypredict, ewt, epsNewt) = {
        let m = IDA_mem.borrow();
        (
            m.ida_yypredict.clone().unwrap(),
            m.ida_ewt.clone().unwrap(),
            m.ida_epsNewt,
        )
    };
    let retval = SUNNonlinSolSolve(
        &NLS,
        &yypredict,
        &ee,
        &ewt,
        epsNewt,
        callLSetup,
        &mut nls_mem,
    );

    /* increment counters */
    let _ = SUNNonlinSolGetNumIters(&NLS, &mut nni_inc);
    IDA_mem.borrow_mut().ida_nni += nni_inc;

    let _ = SUNNonlinSolGetNumConvFails(&NLS, &mut nnf_inc);
    IDA_mem.borrow_mut().ida_nnf += nnf_inc;

    /* return if nonlinear solver failed */
    if retval != sundials_core::sundials_errors::SUN_SUCCESS {
        return retval;
    }

    /* update yy and yp based on the final correction from the nonlinear solver */
    let (yppredict, cj, yy, yp) = {
        let m = IDA_mem.borrow();
        (
            m.ida_yppredict.clone().unwrap(),
            m.ida_cj,
            m.ida_yy.clone().unwrap(),
            m.ida_yp.clone().unwrap(),
        )
    };
    N_VLinearSum(ONE, &yypredict, ONE, &ee, &yy);
    N_VLinearSum(ONE, &yppredict, cj, &ee, &yp);

    IDA_SUCCESS
}

fn IDACheckConstraints(
    IDA_mem: &IDAMem,
    saved_t: sunrealtype,
    step_constraint_fails: &mut i32,
) -> i32 {
    let (mm, tmp, constraints, yy, ewt) = {
        let m = IDA_mem.borrow();
        (
            m.ida_tempv2.clone().unwrap(), /* mask      */
            m.ida_tempv1.clone().unwrap(), /* workspace */
            m.ida_constraints.clone().unwrap(),
            m.ida_yy.clone().unwrap(),
            m.ida_ewt.clone().unwrap(),
        )
    };

    /* Get mask vector mm, 1 where constraints failed and 0 otherwise */
    let constraintsPassed = N_VConstrMask(&constraints, &yy, &mm);
    if constraintsPassed {
        return IDA_SUCCESS;
    }

    /* Constraints not met */

    /* Compute correction v such that y - v will satisfy the constraints
     *
     * 1. Create a mask array that is +1 where constraints are strictly greater
     *    than or less than zero (|c[i]| = 2) and 0 otherwise
     *
     * 2. Create a mask array that is +/- 2 where constraints are strictly greater
     *    than (+) or less than (-) zero and 0 otherwise
     *
     * 3. Use error weights to compute an adjustment vector for values with strict
     *    constraints, a[i] = +/- 2 * w[i] = +/- 2 * (atol * |y[i]| + rtol[i]),
     *    and is 0 otherwise
     *
     * 4. Save the adjustment vector for possible use later
     *
     * 5. Compute correction vector for all values, v[i] = y[i] - 0.1 * a[i] for
     *    strict constraints and v[i] = y[i] otherwise
     *
     * 6. Zero out entries where the constraints passed, v = mask * v
     */
    let tempv3 = IDA_mem.borrow().ida_tempv3.clone().unwrap();
    N_VCompare(ONEPT5, &constraints, &tmp);
    N_VProd(&tmp, &constraints, &tmp);
    N_VDiv(&tmp, &ewt, &tmp);
    N_VScale(-PT1, &tmp, &tempv3);
    N_VLinearSum(ONE, &yy, -PT1, &tmp, &tmp);
    N_VProd(&tmp, &mm, &tmp);

    let vnorm = IDAWrmsNorm(IDA_mem, &tmp, &ewt, SUNFALSE); /* ||v|| */

    /* If constraint correction vector is small in norm (satisfies the nonlinear
    solver convergence condition with R = 1), correct and accept this step */
    if vnorm <= IDA_mem.borrow().ida_epsNewt {
        /* Update constraint correction count */
        IDA_mem.borrow_mut().constraint_corrections += 1;

        /* To reduce roundoff errors that can violate the constraints, split the
         * correction update, ee = ee - v, into three steps */

        let (ee, yypredict) = {
            let m = IDA_mem.borrow();
            (m.ida_ee.clone().unwrap(), m.ida_yypredict.clone().unwrap())
        };

        /* Zero out the correction where any constraint failed */
        N_VProd(&mm, &ee, &tmp);
        N_VLinearSum(ONE, &ee, -ONE, &tmp, &ee);

        /* Set correction to zero out the predictor where any constraint failed */
        N_VProd(&mm, &yypredict, &tmp);
        N_VLinearSum(ONE, &ee, -ONE, &tmp, &ee);

        /* Update the correction where constraints failed and are strictly greater
        or less than zero to shift the state with the adjustment saved above */
        N_VProd(&mm, &tempv3, &tempv3);
        N_VLinearSum(ONE, &ee, -ONE, &tempv3, &ee);

        return IDA_SUCCESS;
    }

    /* update failure counts */
    *step_constraint_fails += 1;
    IDA_mem.borrow_mut().constraint_fails += 1;

    /* Return with error if |h| == hmin */
    {
        let m = IDA_mem.borrow();
        if SUNRabs(m.ida_hh) <= m.ida_hmin * ONEPSM {
            return IDA_CONSTR_FAIL;
        }
    }

    /* Return with error if max step attempt failures */
    if *step_constraint_fails == IDA_mem.borrow().max_constraint_fails {
        return IDA_CONSTR_FAIL;
    }

    /* Constraints correction is too large, reduce h by computing rr = h'/h */
    let phi0 = IDA_mem.borrow().ida_phi[0].clone().unwrap();
    N_VLinearSum(ONE, &phi0, -ONE, &yy, &tmp);
    N_VProd(&mm, &tmp, &tmp);
    let minquot = N_VMinQuotient(&phi0, &tmp);
    {
        let mut m = IDA_mem.borrow_mut();
        m.ida_eta = PT9 * minquot;
        m.ida_eta = SUNMAX(m.ida_eta, PT1);
        m.ida_eta = SUNMAX(m.ida_eta, m.ida_hmin / SUNRabs(m.ida_hh));
    }

    /* Reattempt step with new step size */
    IDARestore(IDA_mem, saved_t);
    {
        let mut m = IDA_mem.borrow_mut();
        m.ida_phase = 1;
        m.ida_hh *= m.ida_eta;
    }
    if IDA_mem.borrow().ida_nst == 0 {
        IDAReset(IDA_mem);
    }

    PREDICT_AGAIN
}

/*
 * IDAPredict
 *
 * This routine predicts the new values for vectors yy and yp.
 */

fn IDAPredict(IDA_mem: &IDAMem) {
    let (kk, cvals, gvals, phivecs, phivecs1, yypredict, yppredict) = {
        let mut m = IDA_mem.borrow_mut();
        let kk = m.ida_kk as usize;

        for j in 0..=kk {
            m.ida_cvals[j] = ONE;
        }

        let cvals: Vec<sunrealtype> = m.ida_cvals[0..=kk].to_vec();
        /* C: `ida_gamma + 1` / `ida_phi + 1` pointer offsets */
        let gvals: Vec<sunrealtype> = m.ida_gamma[1..=kk].to_vec();
        let phivecs: Vec<N_Vector> = (0..=kk).map(|j| m.ida_phi[j].clone().unwrap()).collect();
        let phivecs1: Vec<N_Vector> = (1..=kk).map(|j| m.ida_phi[j].clone().unwrap()).collect();

        (
            kk,
            cvals,
            gvals,
            phivecs,
            phivecs1,
            m.ida_yypredict.clone().unwrap(),
            m.ida_yppredict.clone().unwrap(),
        )
    };

    let _ = N_VLinearCombination((kk + 1) as i32, &cvals, &phivecs, &yypredict);

    let _ = N_VLinearCombination(kk as i32, &gvals, &phivecs1, &yppredict);
}

/*
 * -----------------------------------------------------------------
 * Error test
 * -----------------------------------------------------------------
 */

/*
 * IDATestError
 *
 * This routine estimates errors at orders k, k-1, k-2, decides
 * whether or not to suggest an order decrease, and performs
 * the local error test.
 *
 * IDATestError returns either IDA_SUCCESS or ERROR_TEST_FAIL.
 */

fn IDATestError(
    IDA_mem: &IDAMem,
    ck: sunrealtype,
    err_k: &mut sunrealtype,
    err_km1: &mut sunrealtype,
) -> i32 {
    /* Compute error for order k. */
    let (ee, ewt, suppressalg, kk) = {
        let m = IDA_mem.borrow();
        (
            m.ida_ee.clone().unwrap(),
            m.ida_ewt.clone().unwrap(),
            m.ida_suppressalg,
            m.ida_kk,
        )
    };
    let enorm_k = IDAWrmsNorm(IDA_mem, &ee, &ewt, suppressalg);
    *err_k = IDA_mem.borrow().ida_sigma[kk as usize] * enorm_k;
    let terr_k = (kk + 1) as sunrealtype * (*err_k);

    IDA_mem.borrow_mut().ida_knew = kk;

    if kk > 1 {
        /* Compute error at order k-1 */
        let (phi_kk, delta) = {
            let m = IDA_mem.borrow();
            (
                m.ida_phi[kk as usize].clone().unwrap(),
                m.ida_delta.clone().unwrap(),
            )
        };
        N_VLinearSum(ONE, &phi_kk, ONE, &ee, &delta);
        let enorm_km1 = IDAWrmsNorm(IDA_mem, &delta, &ewt, suppressalg);
        *err_km1 = IDA_mem.borrow().ida_sigma[(kk - 1) as usize] * enorm_km1;
        let terr_km1 = kk as sunrealtype * (*err_km1);

        if kk > 2 {
            /* Compute error at order k-2 */
            let phi_km1 = IDA_mem.borrow().ida_phi[(kk - 1) as usize].clone().unwrap();
            N_VLinearSum(ONE, &phi_km1, ONE, &delta, &delta);
            let enorm_km2 = IDAWrmsNorm(IDA_mem, &delta, &ewt, suppressalg);
            let err_km2 = IDA_mem.borrow().ida_sigma[(kk - 2) as usize] * enorm_km2;
            let terr_km2 = (kk - 1) as sunrealtype * err_km2;

            /* Decrease order if errors are reduced */
            if SUNMAX(terr_km1, terr_km2) <= terr_k {
                IDA_mem.borrow_mut().ida_knew = kk - 1;
            }
        } else {
            /* Decrease order to 1 if errors are reduced by at least 1/2 */
            if terr_km1 <= (HALF * terr_k) {
                IDA_mem.borrow_mut().ida_knew = kk - 1;
            }
        }
    }

    /* Perform error test */
    if ck * enorm_k > ONE {
        ERROR_TEST_FAIL
    } else {
        IDA_SUCCESS
    }
}

/*
 * IDARestore
 *
 * This routine restores tn, psi, and phi in the event of a failure.
 * It changes back phi-star to phi (changed in IDASetCoeffs)
 */

fn IDARestore(IDA_mem: &IDAMem, saved_t: sunrealtype) {
    let (ns, kk) = {
        let mut m = IDA_mem.borrow_mut();

        m.ida_tn = saved_t;

        let kk = m.ida_kk as usize;
        for j in 1..=kk {
            m.ida_psi[j - 1] = m.ida_psi[j] - m.ida_hh;
        }

        (m.ida_ns, m.ida_kk)
    };

    if ns <= kk {
        let ns = ns as usize;
        let kk = kk as usize;
        let (cvals, phivecs) = {
            let mut m = IDA_mem.borrow_mut();
            for j in ns..=kk {
                m.ida_cvals[j - ns] = ONE / m.ida_beta[j];
            }
            let cvals: Vec<sunrealtype> = m.ida_cvals[0..=(kk - ns)].to_vec();
            let phivecs: Vec<N_Vector> = (ns..=kk).map(|j| m.ida_phi[j].clone().unwrap()).collect();
            (cvals, phivecs)
        };
        /* Same slice for X and Z: selects the in-place branch, as in C where
        both arguments are the `ida_phi + ns` pointer. */
        let _ = N_VScaleVectorArray((kk - ns + 1) as i32, &cvals, &phivecs, &phivecs);
    }
}

/*
 * -----------------------------------------------------------------
 * Handler for convergence and/or error test failures
 * -----------------------------------------------------------------
 */

/*
 * IDAHandleNFlag
 *
 * This routine handles failures indicated by the input variable nflag.
 * Positive values indicate various recoverable failures while negative
 * values indicate nonrecoverable failures. This routine adjusts the
 * step size for recoverable failures.
 *
 *  Possible nflag values (input):
 *
 *   --convergence failures--
 *   IDA_RES_RECVR              > 0
 *   IDA_LSOLVE_RECVR           > 0
 *   SUN_NLS_CONV_RECVR         > 0
 *   IDA_RES_FAIL               < 0
 *   IDA_LSOLVE_FAIL            < 0
 *   IDA_LSETUP_FAIL            < 0
 *
 *   --error test failure--
 *   ERROR_TEST_FAIL            > 0
 *
 *  Possible kflag values (output):
 *
 *   --recoverable--
 *   PREDICT_AGAIN
 *
 *   --nonrecoverable--
 *   IDA_REP_RES_ERR
 *   IDA_ERR_FAIL
 *   IDA_CONV_FAIL
 *   IDA_RES_FAIL
 *   IDA_LSETUP_FAIL
 *   IDA_LSOLVE_FAIL
 *
 * NOTE (port): C also takes `long int* ncfnPtr` and `long int* netfPtr`.
 * The single call site (IDAStep) always passes `&IDA_mem->ida_ncfn` and
 * `&IDA_mem->ida_netf`; a Rust `&mut` into those RefCell fields cannot
 * coexist with the `&IDAMem` argument, so the global counters are
 * incremented through the mem handle at exactly the same points.
 */

fn IDAHandleNFlag(
    IDA_mem: &IDAMem,
    nflag: i32,
    err_k: sunrealtype,
    err_km1: sunrealtype,
    ncfPtr: &mut i32,
    nefPtr: &mut i32,
) -> i32 {
    IDA_mem.borrow_mut().ida_phase = 1;

    if nflag != ERROR_TEST_FAIL {
        /*-----------------------
        Nonlinear solver failed
        -----------------------*/

        *ncfPtr += 1; /* local counter for convergence failures */
        IDA_mem.borrow_mut().ida_ncfn += 1; /* global counter (C: *ncfnPtr) */

        if nflag < 0 {
            /* nonrecoverable failure */

            if nflag == IDA_LSOLVE_FAIL {
                IDA_LSOLVE_FAIL
            } else if nflag == IDA_LSETUP_FAIL {
                IDA_LSETUP_FAIL
            } else if nflag == IDA_RES_FAIL {
                IDA_RES_FAIL
            } else {
                IDA_NLS_FAIL
            }
        } else {
            /* recoverable failure    */

            /* Test if there were too many convergence failures or |h| = hmin */
            {
                let m = IDA_mem.borrow();
                if (*ncfPtr == m.ida_maxncf) || (SUNRabs(m.ida_hh) <= m.ida_hmin * ONEPSM) {
                    if nflag == IDA_RES_RECVR {
                        return IDA_REP_RES_ERR;
                    }
                    return IDA_CONV_FAIL;
                }
            }

            /* Reduce step size for a new prediction */
            {
                let mut m = IDA_mem.borrow_mut();
                m.ida_eta = SUNMAX(m.ida_eta_cf, m.ida_hmin / SUNRabs(m.ida_hh));
                m.ida_hh *= m.ida_eta;
            }

            PREDICT_AGAIN
        }
    } else {
        /*-----------------
        Error Test failed
        -----------------*/

        *nefPtr += 1; /* local counter for error test failures */
        IDA_mem.borrow_mut().ida_netf += 1; /* global counter (C: *netfPtr) */

        if *nefPtr == 1 {
            /* On first error test failure, keep current order or lower order by one.
            Compute new stepsize based on differences of the solution. */

            let mut m = IDA_mem.borrow_mut();
            let err_knew = if m.ida_kk == m.ida_knew {
                err_k
            } else {
                err_km1
            };

            m.ida_kk = m.ida_knew;
            m.ida_eta = PT9
                * SUNRpowerR(
                    TWO * err_knew + PT0001,
                    -ONE / ((m.ida_kk + 1) as sunrealtype),
                );
            m.ida_eta = SUNMAX(m.ida_eta_min_ef, SUNMIN(m.ida_eta_low, m.ida_eta));
            m.ida_eta = SUNMAX(m.ida_eta, m.ida_hmin / SUNRabs(m.ida_hh));
            m.ida_hh *= m.ida_eta;

            PREDICT_AGAIN
        } else if *nefPtr == 2 {
            /* On second error test failure, use current order or decrease order by one.
            Reduce stepsize by factor of 1/4. */

            let mut m = IDA_mem.borrow_mut();
            m.ida_kk = m.ida_knew;
            m.ida_eta = SUNMAX(m.ida_eta_min_ef, m.ida_hmin / SUNRabs(m.ida_hh));
            m.ida_hh *= m.ida_eta;

            PREDICT_AGAIN
        } else if *nefPtr < IDA_mem.borrow().ida_maxnef {
            /* On third and subsequent error test failures, set order to 1.
            Reduce stepsize by factor of 1/4. */
            let mut m = IDA_mem.borrow_mut();
            m.ida_kk = 1;
            m.ida_eta = SUNMAX(m.ida_eta_min_ef, m.ida_hmin / SUNRabs(m.ida_hh));
            m.ida_hh *= m.ida_eta;

            PREDICT_AGAIN
        } else {
            /* Too many error test failures */
            IDA_ERR_FAIL
        }
    }
}

/*
 * IDAReset
 *
 * This routine is called only if we need to predict again at the
 * very first step. In such a case, reset phi[1] and psi[0].
 */

fn IDAReset(IDA_mem: &IDAMem) {
    let (eta, phi1) = {
        let mut m = IDA_mem.borrow_mut();
        m.ida_psi[0] = m.ida_hh;
        (m.ida_eta, m.ida_phi[1].clone().unwrap())
    };

    N_VScale(eta, &phi1, &phi1);
}

/*
 * -----------------------------------------------------------------
 * Function called after a successful step
 * -----------------------------------------------------------------
 */

/*
 * IDACompleteStep
 *
 * This routine completes a successful step.  It increments nst,
 * saves the stepsize and order used, makes the final selection of
 * stepsize and order for the next step, and updates the phi array.
 */

fn IDACompleteStep(IDA_mem: &IDAMem, err_k: sunrealtype, err_km1: sunrealtype) {
    let kdiff;
    let phase;
    {
        let mut m = IDA_mem.borrow_mut();

        m.ida_nst += 1;
        kdiff = m.ida_kk - m.ida_kused;
        m.ida_kused = m.ida_kk;
        m.ida_hused = m.ida_hh;

        if (m.ida_knew == m.ida_kk - 1) || (m.ida_kk == m.ida_maxord) {
            m.ida_phase = 1;
        }

        phase = m.ida_phase;
    }

    /* For the first few steps, until either a step fails, or the order is
    reduced, or the order reaches its maximum, we raise the order and double
    the stepsize. During these steps, phase = 0. Thereafter, phase = 1, and
    stepsize and order are set by the usual local error algorithm.

    Note that, after the first step, the order is not increased, as not all
    of the necessary information is available yet. */

    if phase == 0 {
        let mut m = IDA_mem.borrow_mut();
        if m.ida_nst > 1 {
            m.ida_kk += 1;
            let mut hnew = TWO * m.ida_hh;
            let tmp = SUNRabs(hnew) * m.ida_hmax_inv;
            if tmp > ONE {
                hnew /= tmp;
            }
            m.ida_hh = hnew;
        }
    } else {
        /* (C initializes action = UNSET here; every path below assigns action
        before it is read, so the dead store is omitted.) */
        let action: i32;

        /* C leaves err_kp1 uninitialized; it is written only in the branch that
        is also the only one able to select action = RAISE, so this initializer
        is never observable. */
        let mut err_kp1 = ZERO;

        let (knew, kk, maxord, ns) = {
            let m = IDA_mem.borrow();
            (m.ida_knew, m.ida_kk, m.ida_maxord, m.ida_ns)
        };

        /* Set action = LOWER/MAINTAIN/RAISE to specify order decision */

        if knew == kk - 1 {
            /* Already decided to reduce the order */
            action = LOWER;
        } else if kk == maxord {
            /* Already using the maximum order */
            action = MAINTAIN;
        } else if (kk + 1 >= ns) || (kdiff == 1) {
            /* Step size has not been constant or the order was just raised */
            action = MAINTAIN;
        } else {
            /* Estimate the error at order k+1 */

            let (ee, phi_kp1, tempv1, ewt, suppressalg) = {
                let m = IDA_mem.borrow();
                (
                    m.ida_ee.clone().unwrap(),
                    m.ida_phi[(kk + 1) as usize].clone().unwrap(),
                    m.ida_tempv1.clone().unwrap(),
                    m.ida_ewt.clone().unwrap(),
                    m.ida_suppressalg,
                )
            };
            N_VLinearSum(ONE, &ee, -ONE, &phi_kp1, &tempv1);
            let enorm = IDAWrmsNorm(IDA_mem, &tempv1, &ewt, suppressalg);
            err_kp1 = enorm / ((kk + 2) as sunrealtype);

            /* Choose among orders k-1, k, k+1 using local truncation error norms. */

            let terr_k = (kk + 1) as sunrealtype * err_k;
            let terr_kp1 = (kk + 2) as sunrealtype * err_kp1;

            if kk == 1 {
                if terr_kp1 >= HALF * terr_k {
                    action = MAINTAIN;
                } else {
                    action = RAISE;
                }
            } else {
                let terr_km1 = kk as sunrealtype * err_km1;
                if terr_km1 <= SUNMIN(terr_k, terr_kp1) {
                    action = LOWER;
                } else if terr_kp1 >= terr_k {
                    action = MAINTAIN;
                } else {
                    action = RAISE;
                }
            }
        }

        /* Set the estimated error norm and, on change of order, reset kk. */
        let err_knew;
        if action == RAISE {
            IDA_mem.borrow_mut().ida_kk += 1;
            err_knew = err_kp1;
        } else if action == LOWER {
            IDA_mem.borrow_mut().ida_kk -= 1;
            err_knew = err_km1;
        } else {
            err_knew = err_k;
        }

        /* Compute tmp = tentative ratio hnew/hh from error norm estimate.
        1. If eta >= eta_max_fx (default = 2), increase hh to at most eta_max
           (default = 2) i.e., double the step size
        2. If eta <= eta_min_fx (default = 1), reduce hh to between eta_min
           (default 0.5) and eta_low (default 0.9),
        3. Otherwise leave hh as is i.e., eta = 1. */

        {
            let mut m = IDA_mem.borrow_mut();

            m.ida_eta = ONE;
            let tmp = SUNRpowerR(
                TWO * err_knew + PT0001,
                -ONE / ((m.ida_kk + 1) as sunrealtype),
            );

            if tmp >= m.ida_eta_max_fx {
                /* Enforce max growth factor bound and max step size */
                m.ida_eta = SUNMIN(tmp, m.ida_eta_max);
                let bound = SUNMAX(ONE, m.ida_eta * SUNRabs(m.ida_hh) * m.ida_hmax_inv);
                m.ida_eta /= bound;
            } else if tmp <= m.ida_eta_min_fx {
                /* Enforce required reduction factor bound, min reduction bound, and min
                step size. Note if eta = eta_min_fx = 1 and ida_eta_low < 1 the step
                size is reduced. */
                m.ida_eta = SUNMIN(tmp, m.ida_eta_low);
                m.ida_eta = SUNMAX(m.ida_eta, m.ida_eta_min);
                m.ida_eta = SUNMAX(m.ida_eta, m.ida_hmin / SUNRabs(m.ida_hh));
            }
            m.ida_hh *= m.ida_eta;
        }
    } /* end of phase if block */

    /* Save ee for possible order increase on next step */
    {
        let (kused, maxord) = {
            let m = IDA_mem.borrow();
            (m.ida_kused, m.ida_maxord)
        };
        if kused < maxord {
            let (ee, phi_next) = {
                let m = IDA_mem.borrow();
                (
                    m.ida_ee.clone().unwrap(),
                    m.ida_phi[(kused + 1) as usize].clone().unwrap(),
                )
            };
            N_VScale(ONE, &ee, &phi_next);
        }
    }

    /* Update phi arrays */

    /* To update phi arrays compute X += Z where                  */
    /* X = [ phi[kused], phi[kused-1], phi[kused-2], ... phi[1] ] */
    /* Z = [ ee,         phi[kused],   phi[kused-1], ... phi[0] ] */

    let (nvec, Xvecs, Zvecs) = {
        let mut m = IDA_mem.borrow_mut();
        let kused = m.ida_kused as usize;

        let mut Zvecs: Vec<N_Vector> = Vec::with_capacity(kused + 1);
        let mut Xvecs: Vec<N_Vector> = Vec::with_capacity(kused + 1);
        Zvecs.push(m.ida_ee.clone().unwrap());
        Xvecs.push(m.ida_phi[kused].clone().unwrap());
        for j in 1..=kused {
            Zvecs.push(m.ida_phi[kused - j + 1].clone().unwrap());
            Xvecs.push(m.ida_phi[kused - j].clone().unwrap());
        }

        /* mirror the C mem state (ida_Xvecs / ida_Zvecs are pure scratch) */
        m.ida_Xvecs = Xvecs.clone();
        m.ida_Zvecs = Zvecs.clone();

        ((kused + 1) as i32, Xvecs, Zvecs)
    };

    /* C passes `ida_Xvecs` for both X and Z; the port passes the same slice so
    the array-pointer equality test selects the identical in-place (axpy)
    branch, preserving the sequential cascade over the phi columns. */
    let _ = N_VLinearSumVectorArray(nvec, ONE, &Xvecs, ONE, &Zvecs, &Xvecs);
}

/*
 * -----------------------------------------------------------------
 * Interpolated output
 * -----------------------------------------------------------------
 */

/*
 * IDAGetSolution
 *
 * This routine evaluates y(t) and y'(t) as the value and derivative of
 * the interpolating polynomial at the independent variable t, and stores
 * the results in the vectors yret and ypret.  It uses the current
 * independent variable value, tn, and the method order last used, kused.
 * This function is called by IDASolve with t = tout, t = tn, or t = tstop.
 *
 * If kused = 0 (no step has been taken), or if t = tn, then the order used
 * here is taken to be 1, giving yret = phi[0], ypret = phi[1]/psi[0].
 *
 * The return values are:
 *   IDA_SUCCESS  if t is legal, or
 *   IDA_BAD_T    if t is not within the interval of the last step taken.
 */

pub fn IDAGetSolution(ida_mem: &IDAMem, t: sunrealtype, yret: &N_Vector, ypret: &N_Vector) -> i32 {
    /* NULL-mem check: handled by the type system */
    let IDA_mem = ida_mem;

    /* Check t for legality.  Here tn - hused is t_{n-1}. */

    let (uround, tn, hh, hused, kused) = {
        let m = IDA_mem.borrow();
        (m.ida_uround, m.ida_tn, m.ida_hh, m.ida_hused, m.ida_kused)
    };

    let mut tfuzz = HUNDRED * uround * (SUNRabs(tn) + SUNRabs(hh));
    if hh < ZERO {
        tfuzz = -tfuzz;
    }
    let tp = tn - hused - tfuzz;
    if (t - tp) * hh < ZERO {
        IDAProcessError(
            Some(IDA_mem),
            IDA_BAD_T,
            line!() as i32,
            "IDAGetSolution",
            file!(),
            &MSG_BAD_T(t, tn - hused, tn),
        );
        return IDA_BAD_T;
    }

    /* Initialize kord = (kused or 1). */

    let mut kord = kused;
    if kused == 0 {
        kord = 1;
    }

    /* Accumulate multiples of columns phi[j] into yret and ypret. */

    let (kord, cvals, dvals, phivecs, phivecs1) = {
        let mut m = IDA_mem.borrow_mut();

        let delt = t - m.ida_tn;
        let mut c = ONE;
        let mut d = ZERO;
        let mut gam = delt / m.ida_psi[0];

        let kord = kord as usize;

        m.ida_cvals[0] = c;
        for j in 1..=kord {
            d = d * gam + c / m.ida_psi[j - 1];
            c = c * gam;
            gam = (delt + m.ida_psi[j - 1]) / m.ida_psi[j];

            m.ida_cvals[j] = c;
            m.ida_dvals[j - 1] = d;
        }

        let cvals: Vec<sunrealtype> = m.ida_cvals[0..=kord].to_vec();
        let dvals: Vec<sunrealtype> = m.ida_dvals[0..kord].to_vec();
        let phivecs: Vec<N_Vector> = (0..=kord).map(|j| m.ida_phi[j].clone().unwrap()).collect();
        /* C: `ida_phi + 1` pointer offset */
        let phivecs1: Vec<N_Vector> = (1..=kord).map(|j| m.ida_phi[j].clone().unwrap()).collect();

        (kord, cvals, dvals, phivecs, phivecs1)
    };

    let retval = N_VLinearCombination((kord + 1) as i32, &cvals, &phivecs, yret);
    if retval != IDA_SUCCESS {
        return IDA_VECTOROP_ERR;
    }

    let retval = N_VLinearCombination(kord as i32, &dvals, &phivecs1, ypret);
    if retval != IDA_SUCCESS {
        return IDA_VECTOROP_ERR;
    }

    IDA_SUCCESS
}

/*
 * -----------------------------------------------------------------
 * Norm function
 * -----------------------------------------------------------------
 */

/*
 * IDAWrmsNorm
 *
 *  Returns the WRMS norm of vector x with weights w.
 *  If mask = SUNTRUE, the weight vector w is masked by id, i.e.,
 *      nrm = N_VWrmsNormMask(x,w,id);
 *  Otherwise,
 *      nrm = N_VWrmsNorm(x,w);
 *
 * mask = SUNFALSE       when the call is made from the nonlinear solver.
 * mask = suppressalg otherwise.
 */

pub fn IDAWrmsNorm(
    IDA_mem: &IDAMem,
    x: &N_Vector,
    w: &N_Vector,
    mask: sunbooleantype,
) -> sunrealtype {
    let nrm;

    if mask {
        let id = IDA_mem.borrow().ida_id.clone().unwrap();
        nrm = N_VWrmsNormMask(x, w, &id);
    } else {
        nrm = N_VWrmsNorm(x, w);
    }

    nrm
}

/*
 * -----------------------------------------------------------------
 * Functions for rootfinding
 * -----------------------------------------------------------------
 */

/*
 * IDARcheck1
 *
 * This routine completes the initialization of rootfinding memory
 * information, and checks whether g has a zero both at and very near
 * the initial point of the IVP.
 *
 * This routine returns an int equal to:
 *  IDA_RTFUNC_FAIL < 0 if the g function failed, or
 *  IDA_SUCCESS     = 0 otherwise.
 */

fn IDARcheck1(IDA_mem: &IDAMem) -> i32 {
    {
        let mut m = IDA_mem.borrow_mut();
        for i in 0..m.ida_nrtfn as usize {
            m.ida_iroots[i] = 0;
        }
        m.ida_tlo = m.ida_tn;
        m.ida_ttol = (SUNRabs(m.ida_tn) + SUNRabs(m.ida_hh)) * m.ida_uround * HUNDRED;
    }

    /* Evaluate g at initial t and check for zero values. */
    let (tlo, phi0, phi1) = {
        let m = IDA_mem.borrow();
        (
            m.ida_tlo,
            m.ida_phi[0].clone().unwrap(),
            m.ida_phi[1].clone().unwrap(),
        )
    };
    let gfun = IDA_mem.borrow().ida_gfun.expect("ida_gfun set");
    let mut glo = std::mem::take(&mut IDA_mem.borrow_mut().ida_glo);
    let mut user_data = IDA_mem.borrow_mut().ida_user_data.take();
    let retval = gfun(tlo, &phi0, &phi1, &mut glo, &mut user_data);
    {
        let mut m = IDA_mem.borrow_mut();
        m.ida_user_data = user_data;
        m.ida_glo = glo;
        m.ida_nge = 1;
    }
    if retval != 0 {
        return IDA_RTFUNC_FAIL;
    }

    let mut zroot = SUNFALSE;
    {
        let mut m = IDA_mem.borrow_mut();
        for i in 0..m.ida_nrtfn as usize {
            if SUNRabs(m.ida_glo[i]) == ZERO {
                zroot = SUNTRUE;
                m.ida_gactive[i] = SUNFALSE;
            }
        }
    }
    if !zroot {
        return IDA_SUCCESS;
    }

    /* Some g_i is zero at t0; look at g at t0+(small increment). */
    let (smallh, tplus, phi0, phi1, yy) = {
        let m = IDA_mem.borrow();
        let hratio = SUNMAX(m.ida_ttol / SUNRabs(m.ida_hh), PT1);
        let smallh = hratio * m.ida_hh;
        let tplus = m.ida_tlo + smallh;
        (
            smallh,
            tplus,
            m.ida_phi[0].clone().unwrap(),
            m.ida_phi[1].clone().unwrap(),
            m.ida_yy.clone().unwrap(),
        )
    };
    N_VLinearSum(ONE, &phi0, smallh, &phi1, &yy);
    let gfun = IDA_mem.borrow().ida_gfun.expect("ida_gfun set");
    let mut ghi = std::mem::take(&mut IDA_mem.borrow_mut().ida_ghi);
    let mut user_data = IDA_mem.borrow_mut().ida_user_data.take();
    let retval = gfun(tplus, &yy, &phi1, &mut ghi, &mut user_data);
    {
        let mut m = IDA_mem.borrow_mut();
        m.ida_user_data = user_data;
        m.ida_ghi = ghi;
        m.ida_nge += 1;
    }
    if retval != 0 {
        return IDA_RTFUNC_FAIL;
    }

    /* We check now only the components of g which were exactly 0.0 at t0
     * to see if we can 'activate' them. */
    {
        let mut m = IDA_mem.borrow_mut();
        for i in 0..m.ida_nrtfn as usize {
            if !m.ida_gactive[i] && SUNRabs(m.ida_ghi[i]) != ZERO {
                m.ida_gactive[i] = SUNTRUE;
                m.ida_glo[i] = m.ida_ghi[i];
            }
        }
    }
    IDA_SUCCESS
}

/*
 * IDARcheck2
 *
 * This routine checks for exact zeros of g at the last root found,
 * if the last return was a root.  It then checks for a close pair of
 * zeros (an error condition), and for a new root at a nearby point.
 * The array glo = g(tlo) at the left endpoint of the search interval
 * is adjusted if necessary to assure that all g_i are nonzero
 * there, before returning to do a root search in the interval.
 *
 * On entry, tlo = tretlast is the last value of tret returned by
 * IDASolve.  This may be the previous tn, the previous tout value,
 * or the last root location.
 *
 * This routine returns an int equal to:
 *     IDA_RTFUNC_FAIL < 0 if the g function failed, or
 *     CLOSERT         = 3 if a close pair of zeros was found, or
 *     RTFOUND         = 1 if a new zero of g was found near tlo, or
 *     IDA_SUCCESS     = 0 otherwise.
 */

fn IDARcheck2(IDA_mem: &IDAMem) -> i32 {
    if IDA_mem.borrow().ida_irfnd == 0 {
        return IDA_SUCCESS;
    }

    let (tlo, yy, yp) = {
        let m = IDA_mem.borrow();
        (
            m.ida_tlo,
            m.ida_yy.clone().unwrap(),
            m.ida_yp.clone().unwrap(),
        )
    };
    let _ = IDAGetSolution(IDA_mem, tlo, &yy, &yp);
    let gfun = IDA_mem.borrow().ida_gfun.expect("ida_gfun set");
    let mut glo = std::mem::take(&mut IDA_mem.borrow_mut().ida_glo);
    let mut user_data = IDA_mem.borrow_mut().ida_user_data.take();
    let retval = gfun(tlo, &yy, &yp, &mut glo, &mut user_data);
    {
        let mut m = IDA_mem.borrow_mut();
        m.ida_user_data = user_data;
        m.ida_glo = glo;
        m.ida_nge += 1;
    }
    if retval != 0 {
        return IDA_RTFUNC_FAIL;
    }

    let mut zroot = SUNFALSE;
    {
        let mut m = IDA_mem.borrow_mut();
        for i in 0..m.ida_nrtfn as usize {
            m.ida_iroots[i] = 0;
        }
        for i in 0..m.ida_nrtfn as usize {
            if !m.ida_gactive[i] {
                continue;
            }
            if SUNRabs(m.ida_glo[i]) == ZERO {
                zroot = SUNTRUE;
                m.ida_iroots[i] = 1;
            }
        }
    }
    if !zroot {
        return IDA_SUCCESS;
    }

    /* One or more g_i has a zero at tlo.  Check g at tlo+smallh. */
    let (smallh, tplus, before_tn) = {
        let mut m = IDA_mem.borrow_mut();
        m.ida_ttol = (SUNRabs(m.ida_tn) + SUNRabs(m.ida_hh)) * m.ida_uround * HUNDRED;
        let smallh = if m.ida_hh > ZERO {
            m.ida_ttol
        } else {
            -m.ida_ttol
        };
        let tplus = m.ida_tlo + smallh;
        (smallh, tplus, (tplus - m.ida_tn) * m.ida_hh >= ZERO)
    };
    if before_tn {
        let (hratio, phi1) = {
            let m = IDA_mem.borrow();
            (smallh / m.ida_hh, m.ida_phi[1].clone().unwrap())
        };
        N_VLinearSum(ONE, &yy, hratio, &phi1, &yy);
    } else {
        let _ = IDAGetSolution(IDA_mem, tplus, &yy, &yp);
    }
    let gfun = IDA_mem.borrow().ida_gfun.expect("ida_gfun set");
    let mut ghi = std::mem::take(&mut IDA_mem.borrow_mut().ida_ghi);
    let mut user_data = IDA_mem.borrow_mut().ida_user_data.take();
    let retval = gfun(tplus, &yy, &yp, &mut ghi, &mut user_data);
    {
        let mut m = IDA_mem.borrow_mut();
        m.ida_user_data = user_data;
        m.ida_ghi = ghi;
        m.ida_nge += 1;
    }
    if retval != 0 {
        return IDA_RTFUNC_FAIL;
    }

    /* Check for close roots (error return), for a new zero at tlo+smallh,
    and for a g_i that changed from zero to nonzero. */
    zroot = SUNFALSE;
    {
        let mut m = IDA_mem.borrow_mut();
        for i in 0..m.ida_nrtfn as usize {
            if !m.ida_gactive[i] {
                continue;
            }
            if SUNRabs(m.ida_ghi[i]) == ZERO {
                if m.ida_iroots[i] == 1 {
                    return CLOSERT;
                }
                zroot = SUNTRUE;
                m.ida_iroots[i] = 1;
            } else {
                if m.ida_iroots[i] == 1 {
                    m.ida_glo[i] = m.ida_ghi[i];
                }
            }
        }
    }
    if zroot {
        return RTFOUND;
    }
    IDA_SUCCESS
}

/*
 * IDARcheck3
 *
 * This routine interfaces to IDARootfind to look for a root of g
 * between tlo and either tn or tout, whichever comes first.
 * Only roots beyond tlo in the direction of integration are sought.
 *
 * This routine returns an int equal to:
 *     IDA_RTFUNC_FAIL < 0 if the g function failed, or
 *     RTFOUND         = 1 if a root of g was found, or
 *     IDA_SUCCESS     = 0 otherwise.
 */

fn IDARcheck3(IDA_mem: &IDAMem, tout: sunrealtype, itask: i32) -> i32 {
    /* Set thi = tn or tout, whichever comes first. */
    if itask == IDA_ONE_STEP {
        let mut m = IDA_mem.borrow_mut();
        m.ida_thi = m.ida_tn;
    }
    if itask == IDA_NORMAL {
        let mut m = IDA_mem.borrow_mut();
        let thi = if (tout - m.ida_tn) * m.ida_hh >= ZERO {
            m.ida_tn
        } else {
            tout
        };
        m.ida_thi = thi;
    }

    /* Get y and y' at thi. */
    let (thi, yy, yp) = {
        let m = IDA_mem.borrow();
        (
            m.ida_thi,
            m.ida_yy.clone().unwrap(),
            m.ida_yp.clone().unwrap(),
        )
    };
    let _ = IDAGetSolution(IDA_mem, thi, &yy, &yp);

    /* Set ghi = g(thi) and call IDARootfind to search (tlo,thi) for roots. */
    let gfun = IDA_mem.borrow().ida_gfun.expect("ida_gfun set");
    let mut ghi = std::mem::take(&mut IDA_mem.borrow_mut().ida_ghi);
    let mut user_data = IDA_mem.borrow_mut().ida_user_data.take();
    let retval = gfun(thi, &yy, &yp, &mut ghi, &mut user_data);
    {
        let mut m = IDA_mem.borrow_mut();
        m.ida_user_data = user_data;
        m.ida_ghi = ghi;
        m.ida_nge += 1;
    }
    if retval != 0 {
        return IDA_RTFUNC_FAIL;
    }

    {
        let mut m = IDA_mem.borrow_mut();
        m.ida_ttol = (SUNRabs(m.ida_tn) + SUNRabs(m.ida_hh)) * m.ida_uround * HUNDRED;
    }
    let ier = IDARootfind(IDA_mem);
    if ier == IDA_RTFUNC_FAIL {
        return IDA_RTFUNC_FAIL;
    }
    {
        let mut m = IDA_mem.borrow_mut();
        for i in 0..m.ida_nrtfn as usize {
            if !m.ida_gactive[i] && m.ida_grout[i] != ZERO {
                m.ida_gactive[i] = SUNTRUE;
            }
        }
        m.ida_tlo = m.ida_trout;
        for i in 0..m.ida_nrtfn as usize {
            m.ida_glo[i] = m.ida_grout[i];
        }
    }

    /* If no root found, return IDA_SUCCESS. */
    if ier == IDA_SUCCESS {
        return IDA_SUCCESS;
    }

    /* If a root was found, interpolate to get y(trout) and return.  */
    let trout = IDA_mem.borrow().ida_trout;
    let _ = IDAGetSolution(IDA_mem, trout, &yy, &yp);
    RTFOUND
}

/*
 * IDARootfind
 *
 * This routine solves for a root of g(t) between tlo and thi, if
 * one exists.  Only roots of odd multiplicity (i.e. with a change
 * of sign in one of the g_i), or exact zeros, are found.
 * Here the sign of tlo - thi is arbitrary, but if multiple roots
 * are found, the one closest to tlo is returned.
 *
 * The method used is the Illinois algorithm, a modified secant method.
 * Reference: Kathie L. Hiebert and Lawrence F. Shampine, Implicitly
 * Defined Output Points for Solutions of ODEs, Sandia National
 * Laboratory Report SAND80-0180, February 1980.
 *
 * This routine uses the following parameters for communication:
 *
 * nrtfn    = number of functions g_i, or number of components of
 *            the vector-valued function g(t).  Input only.
 *
 * gfun     = user-defined function for g(t).  Its form is
 *            (void) gfun(t, y, yp, gt, user_data)
 *
 * rootdir  = in array specifying the direction of zero-crossings.
 *            If rootdir[i] > 0, search for roots of g_i only if
 *            g_i is increasing; if rootdir[i] < 0, search for
 *            roots of g_i only if g_i is decreasing; otherwise
 *            always search for roots of g_i.
 *
 * gactive  = array specifying whether a component of g should
 *            or should not be monitored. gactive[i] is initially
 *            set to SUNTRUE for all i=0,...,nrtfn-1, but it may be
 *            reset to SUNFALSE if at the first step g[i] is 0.0
 *            both at the I.C. and at a small perturbation of them.
 *            gactive[i] is then set back on SUNTRUE only after the
 *            corresponding g function moves away from 0.0.
 *
 * nge      = cumulative counter for gfun calls.
 *
 * ttol     = a convergence tolerance for trout.  Input only.
 *            When a root at trout is found, it is located only to
 *            within a tolerance of ttol.  Typically, ttol should
 *            be set to a value on the order of
 *               100 * UROUND * max (SUNRabs(tlo), SUNRabs(thi))
 *            where UROUND is the unit roundoff of the machine.
 *
 * tlo, thi = endpoints of the interval in which roots are sought.
 *            On input, these must be distinct, but tlo - thi may
 *            be of either sign.  The direction of integration is
 *            assumed to be from tlo to thi.  On return, tlo and thi
 *            are the endpoints of the final relevant interval.
 *
 * glo, ghi = arrays of length nrtfn containing the vectors g(tlo)
 *            and g(thi) respectively.  Input and output.  On input,
 *            none of the glo[i] should be zero.
 *
 * trout    = root location, if a root was found, or thi if not.
 *            Output only.  If a root was found other than an exact
 *            zero of g, trout is the endpoint thi of the final
 *            interval bracketing the root, with size at most ttol.
 *
 * grout    = array of length nrtfn containing g(trout) on return.
 *
 * iroots   = int array of length nrtfn with root information.
 *            Output only.  If a root was found, iroots indicates
 *            which components g_i have a root at trout.  For
 *            i = 0, ..., nrtfn-1, iroots[i] = 1 if g_i has a root
 *            and g_i is increasing, iroots[i] = -1 if g_i has a
 *            root and g_i is decreasing, and iroots[i] = 0 if g_i
 *            has no roots or g_i varies in the direction opposite
 *            to that indicated by rootdir[i].
 *
 * This routine returns an int equal to:
 *      IDA_RTFUNC_FAIL < 0 if the g function failed, or
 *      RTFOUND         = 1 if a root of g was found, or
 *      IDA_SUCCESS     = 0 otherwise.
 *
 */

fn IDARootfind(IDA_mem: &IDAMem) -> i32 {
    /* Move the rootfinding state into locals for the duration of the search
    (the user's g function is invoked inside the loop; no RefCell borrow may
    be held across it). C writes through the IDA_mem fields on every return
    path; the single write-back below restores the identical state for each
    path (on the IDA_RTFUNC_FAIL path the fields hold the values from the
    last completed iteration, exactly as in C).
    Take all six together — see accepted deviation B in lib.rs for why a
    partial restore would be worse than this. */
    let (nrtfn, ttol, yy, yp) = {
        let m = IDA_mem.borrow();
        (
            m.ida_nrtfn as usize,
            m.ida_ttol,
            m.ida_yy.clone().unwrap(),
            m.ida_yp.clone().unwrap(),
        )
    };
    let (mut tlo, mut thi, mut trout) = {
        let m = IDA_mem.borrow();
        (m.ida_tlo, m.ida_thi, m.ida_trout)
    };
    let (mut glo, mut ghi, mut grout, mut iroots, rootdir, gactive) = {
        let mut m = IDA_mem.borrow_mut();
        (
            std::mem::take(&mut m.ida_glo),
            std::mem::take(&mut m.ida_ghi),
            std::mem::take(&mut m.ida_grout),
            std::mem::take(&mut m.ida_iroots),
            std::mem::take(&mut m.ida_rootdir),
            std::mem::take(&mut m.ida_gactive),
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
            IDA_SUCCESS if no zero was found, or set iroots and return RTFOUND.  */
            if !sgnchg {
                trout = thi;
                for i in 0..nrtfn {
                    grout[i] = ghi[i];
                }
                if !zroot {
                    return IDA_SUCCESS;
                }
                for i in 0..nrtfn {
                    iroots[i] = 0;
                    if !gactive[i] {
                        continue;
                    }
                    if (SUNRabs(ghi[i]) == ZERO) && (rootdir[i] as sunrealtype * glo[i] <= ZERO) {
                        iroots[i] = if glo[i] > ZERO { -1 } else { 1 };
                    }
                }
                return RTFOUND;
            }

            /* Initialize alph to avoid compiler warning */
            let mut alph = ONE;

            /* A sign change was found.  Loop to locate nearest root. */

            let mut side = 0;
            let mut sideprev = -1;
            loop {
                /* Looping point */

                /* If interval size is already less than tolerance ttol, break. */
                if SUNRabs(thi - tlo) <= ttol {
                    break;
                }

                /* Set weight alph.
                On the first two passes, set alph = 1.  Thereafter, reset alph
                according to the side (low vs high) of the subinterval in which
                the sign change was found in the previous two passes.
                If the sides were opposite, set alph = 1.
                If the sides were the same, then double alph (if high side),
                or halve alph (if low side).
                The next guess tmid is the secant method value if alph = 1, but
                is closer to tlo if alph < 1, and closer to thi if alph > 1.    */

                if sideprev == side {
                    alph = if side == 2 { alph * TWO } else { alph * HALF };
                } else {
                    alph = ONE;
                }

                /* Set next root approximation tmid and get g(tmid).
                If tmid is too close to tlo or thi, adjust it inward,
                by a fractional distance that is between 0.1 and 0.5.  */
                let mut tmid = thi - (thi - tlo) * ghi[imax] / (ghi[imax] - alph * glo[imax]);
                if SUNRabs(tmid - tlo) < HALF * ttol {
                    let fracint = SUNRabs(thi - tlo) / ttol;
                    let fracsub = if fracint > FIVE { PT1 } else { HALF / fracint };
                    tmid = tlo + fracsub * (thi - tlo);
                }
                if SUNRabs(thi - tmid) < HALF * ttol {
                    let fracint = SUNRabs(thi - tlo) / ttol;
                    let fracsub = if fracint > FIVE { PT1 } else { HALF / fracint };
                    tmid = thi - fracsub * (thi - tlo);
                }

                let _ = IDAGetSolution(IDA_mem, tmid, &yy, &yp);
                let gfun = IDA_mem.borrow().ida_gfun.expect("ida_gfun set");
                let mut user_data = IDA_mem.borrow_mut().ida_user_data.take();
                let retval = gfun(tmid, &yy, &yp, &mut grout, &mut user_data);
                {
                    let mut m = IDA_mem.borrow_mut();
                    m.ida_user_data = user_data;
                    m.ida_nge += 1;
                }
                if retval != 0 {
                    return IDA_RTFUNC_FAIL;
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
        let mut m = IDA_mem.borrow_mut();
        m.ida_tlo = tlo;
        m.ida_thi = thi;
        m.ida_trout = trout;
        m.ida_glo = glo;
        m.ida_ghi = ghi;
        m.ida_grout = grout;
        m.ida_iroots = iroots;
        m.ida_rootdir = rootdir;
        m.ida_gactive = gactive;
    }

    retflag
}
