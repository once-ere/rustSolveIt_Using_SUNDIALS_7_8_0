//! Port of `src/idas/idas.c` (+ the public `include/idas/idas.h`
//! declarations, which fold into `idas_impl`).
//!
//! Main IDAS integrator: creation/initialization, quadrature, forward
//! sensitivity and quadrature-sensitivity initialization and tolerance
//! functions, rootfinding initialization, the `IDASolve` driver,
//! `IDAStep` and all its helpers, dense output (`IDAGetDky` and the
//! quadrature/sensitivity variants), the internal error-weight
//! functions, the WRMS norm helpers, rootfinding (`IDARcheck1/2/3`,
//! `IDARootfind`), the internal sensitivity/quadrature-sensitivity DQ
//! residual approximations, and the deallocation functions.
//!
//! `IDAProcessError`, every `MSG_*` message constant/builder, and **all
//! module-scope constants `idas.c` defines** (`ZERO` … `PT0001`,
//! `ONEPSM`, `PREDICT_AGAIN`, `CONTINUE_STEPS`, `UNSET`/`LOWER`/
//! `RAISE`/`MAINTAIN`, `ERROR_TEST_FAIL`, `RTFOUND`/`CLOSERT`,
//! `CENTERED1`/`CENTERED2`/`FORWARD1`/`FORWARD2`, and the algorithmic
//! block `MXNCF`/`MXNEF`/`MAXNH`/`MAXNJ`/`MAXNI`/`EPCON`/`MAXBACKS`)
//! live in `idas_impl` per the frozen fragment-file protocol and reach
//! this module through the `crate::idas_impl::*` glob import below.
//! Fragments of this module must NOT redeclare them.
//!
//! Reference build configuration: SUNDIALS_LOGGING_LEVEL = 2
//! (`SUNLogInfo`/`SUNLogInfoIf`/`SUNLogDebug`/`SUNLogExtraDebug*` call
//! sites omitted entirely at translation time; `IDA_WARNING` paths are
//! kept — they queue through the logger and appear in the reference
//! outputs), profiling off (every `SUNDIALS_MARK_FUNCTION_BEGIN/END`
//! omitted), error checks off (`SUNAssert`/`SUNCheck*` no-ops,
//! `SUNCheckCall` evaluates and continues), monitoring ON, serial
//! branches only.
//!
//! Handle model: `IDAMem = Rc<RefCell<IDAMemRec>>`. Internal functions
//! take `&IDAMem` and use granular borrows — no borrow of the mem is
//! ever held across a user callback, an N_Vector operation on a
//! user-visible vector, an `IDAProcessError` call, or a linear/
//! nonlinear-solver call, all of which can re-enter the mem.
//!
//! `ida_yy`/`ida_yp` alias the user's `yret`/`ypret` (the `Rc` clone
//! shares the underlying data exactly as the C pointer copy does), so
//! the copy-back contract of ARCHITECTURE §"Aliasing / copy-back rule"
//! is satisfied by construction on every `IDASolve` return path.
//!
//! `void*` callback-data protocol (frozen contract §2): `ida_user_dataS`
//! / `ida_user_dataQS` hold `Some(box)` when the box is a module-owned
//! token — an `IDAMem` handle clone, installed whenever the internal DQ
//! routines are in use, exactly as CVODES does for `cv_fS_data` /
//! `cv_fQS_data` — and `None` when C stored the plain `ida_user_data`
//! pointer there, meaning "pass the integrator's `ida_user_data` at call
//! time". Invokers `Option::take` the box, call, and restore it on every
//! path including error returns.

use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

use crate::idas_impl::*;
use sundials_core::sundials_context::SUNContext;
use sundials_core::sundials_math::*;
use sundials_core::sundials_nonlinearsolver::*;
use sundials_core::sundials_nvector::*;
use sundials_core::sundials_types::*;
use sundials_core::sunnonlinsol_newton::{SUNNonlinSol_Newton, SUNNonlinSol_NewtonSens};

/*
 * =================================================================
 * IDAS PRIVATE CONSTANTS / IDAS ROUTINE-SPECIFIC CONSTANTS
 * =================================================================
 *
 * `idas.c` is ported as several concatenated fragments, so **every**
 * module-scope `#define` it declares lives in `idas_impl` (one shared
 * definition — frozen contract §7) and is in scope here through the
 * glob import above:
 *
 *   ZERO HALF TWOTHIRDS ONE ONEPT5 TWO FOUR FIVE TEN TWENTY HUNDRED
 *   PT9 PT1 PT01 PT001 PT0001 ONEPSM
 *   PREDICT_AGAIN CONTINUE_STEPS UNSET LOWER RAISE MAINTAIN
 *   ERROR_TEST_FAIL RTFOUND CLOSERT
 *   CENTERED1 CENTERED2 FORWARD1 FORWARD2
 *   MXNCF MXNEF MAXNH MAXNJ MAXNI EPCON MAXBACKS
 *
 * Nothing from that list may be redefined in any fragment.
 */

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

    /* IDA_mem->ida_sunctx = sunctx (set by zeroed);
    IDA_mem->python = NULL (Python bindings are out of scope) */

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

    /* Set default values for quad. optional inputs */
    IDA_mem.ida_quadr = SUNFALSE;
    IDA_mem.ida_rhsQ = None;
    IDA_mem.ida_errconQ = SUNFALSE;
    IDA_mem.ida_itolQ = IDA_NN;
    IDA_mem.ida_atolQmin0 = SUNTRUE;

    /* Set default values for sensi. optional inputs */
    IDA_mem.ida_sensi = SUNFALSE;
    /* C: ida_user_dataS = (void*)IDA_mem — the module-owned token is
    installed below, once the handle exists */
    IDA_mem.ida_resS = Some(IDASensResDQ);
    IDA_mem.ida_resSDQ = SUNTRUE;
    IDA_mem.ida_DQtype = IDA_CENTERED;
    IDA_mem.ida_DQrhomax = ZERO;
    IDA_mem.ida_p = None;
    IDA_mem.ida_pbar = Vec::new();
    IDA_mem.ida_plist = Vec::new();
    IDA_mem.ida_errconS = SUNFALSE;
    IDA_mem.ida_itolS = IDA_EE;
    IDA_mem.ida_atolSmin0 = Vec::new();
    IDA_mem.ida_ism = -1; /* initialize to invalid option */

    /* Defaults for sensi. quadr. optional inputs. */
    IDA_mem.ida_quadr_sensi = SUNFALSE;
    /* C: ida_user_dataQS = (void*)IDA_mem — see the note above */
    IDA_mem.ida_rhsQS = Some(IDAQuadSensRhsInternalDQ);
    IDA_mem.ida_rhsQSDQ = SUNTRUE;
    IDA_mem.ida_errconQS = SUNFALSE;
    IDA_mem.ida_itolQS = IDA_EE;
    IDA_mem.ida_atolQSmin0 = Vec::new();

    /* Set defaults for ASA. */
    IDA_mem.ida_adj = SUNFALSE;
    IDA_mem.ida_adj_mem = None;

    /* Initialize lrw and liw */
    IDA_mem.ida_lrw = (25 + 5 * MXORDP1) as i64;
    IDA_mem.ida_liw = 38;

    /* No mallocs have been done yet */
    IDA_mem.ida_VatolMallocDone = SUNFALSE;
    IDA_mem.ida_idMallocDone = SUNFALSE;
    IDA_mem.ida_MallocDone = SUNFALSE;

    IDA_mem.ida_VatolQMallocDone = SUNFALSE;
    IDA_mem.ida_quadMallocDone = SUNFALSE;

    IDA_mem.ida_VatolSMallocDone = SUNFALSE;
    IDA_mem.ida_SatolSMallocDone = SUNFALSE;
    IDA_mem.ida_sensMallocDone = SUNFALSE;

    IDA_mem.ida_VatolQSMallocDone = SUNFALSE;
    IDA_mem.ida_SatolQSMallocDone = SUNFALSE;
    IDA_mem.ida_quadSensMallocDone = SUNFALSE;

    IDA_mem.ida_adjMallocDone = SUNFALSE;

    /* Initialize nonlinear solver variables */
    IDA_mem.NLS = None;
    IDA_mem.ownNLS = SUNFALSE;

    IDA_mem.NLSsim = None;
    IDA_mem.ownNLSsim = SUNFALSE;
    IDA_mem.ypredictSim = None;
    IDA_mem.ycorSim = None;
    IDA_mem.ewtSim = None;
    IDA_mem.simMallocDone = SUNFALSE;

    IDA_mem.NLSstg = None;
    IDA_mem.ownNLSstg = SUNFALSE;
    IDA_mem.ypredictStg = None;
    IDA_mem.ycorStg = None;
    IDA_mem.ewtStg = None;
    IDA_mem.stgMallocDone = SUNFALSE;

    let IDA_mem: IDAMem = Rc::new(RefCell::new(IDA_mem));

    /* C: IDA_mem->ida_user_dataS = IDA_mem->ida_user_dataQS =
    (void*)IDA_mem. The port stores an IDAMem handle CLONE inside the
    Box<dyn Any> token (frozen contract §2); the DQ residual routines
    downcast it back to `IDAMem`. */
    let tokenS: Box<dyn Any> = Box::new(IDA_mem.clone());
    let tokenQS: Box<dyn Any> = Box::new(IDA_mem.clone());
    {
        let mut m = IDA_mem.borrow_mut();
        m.ida_user_dataS = Some(tokenS);
        m.ida_user_dataQS = Some(tokenQS);
    }

    /* Return pointer to IDA memory block */
    Some(IDA_mem)
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

    /* Allocate temporary work arrays for fused vector ops. The C code
    mallocs MXORDP1 slots for cvals/Xvecs/Zvecs; `ida_Xvecs`/`ida_Zvecs`
    are handle scratch that callers rebuild on demand (an N_Vector array
    cannot be left uninitialized in safe Rust), so only `ida_cvals` is
    materialized. The C NULL-check failure branch (which would call
    IDAFreeVectors and report MSG_MEM_FAIL) is unreachable: Vec
    allocation aborts rather than returning NULL. */
    {
        let mut m = IDA_mem.borrow_mut();
        m.ida_cvals = vec![ZERO; MXORDP1];
        m.ida_Xvecs = Vec::new();
        m.ida_Zvecs = Vec::new();
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
    let retval = crate::idas_nls::IDASetNonlinearSolver(IDA_mem, &NLS);

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

        /* Set forceSetup to SUNFALSE */

        m.ida_forceSetup = SUNFALSE;

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

    /* Set forceSetup to SUNFALSE */

    IDA_mem.borrow_mut().ida_forceSetup = SUNFALSE;

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
        let _ =
            crate::idas_ls::idaLsInitializeCounters(&mut crate::idas_ls::idals_mem_mut(IDA_mem));
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
 * IDAQuadMalloc
 *
 * IDAQuadMalloc allocates and initializes quadrature related
 * memory for a problem. All problem specification inputs are
 * checked for errors. If any error occurs during initialization,
 * it is reported to the file whose file pointer is errfp.
 * The return value is IDA_SUCCESS = 0 if no errors occurred, or
 * a negative value otherwise.
 */

pub fn IDAQuadInit(ida_mem: &IDAMem, rhsQ: IDAQuadRhsFn, yQ0: &N_Vector) -> i32 {
    /* Check ida_mem */
    /* NULL-mem check: handled by type system */
    let IDA_mem = ida_mem;

    /* Set space requirements for one N_Vector */
    let mut lrw1Q: sunindextype = 0;
    let mut liw1Q: sunindextype = 0;
    N_VSpace(yQ0, &mut lrw1Q, &mut liw1Q);
    {
        let mut m = IDA_mem.borrow_mut();
        m.ida_lrw1Q = lrw1Q;
        m.ida_liw1Q = liw1Q;
    }

    /* Allocate the vectors (using yQ0 as a template) */
    let allocOK = IDAQuadAllocVectors(IDA_mem, yQ0);
    if !allocOK {
        IDAProcessError(
            Some(IDA_mem),
            IDA_MEM_FAIL,
            line!() as i32,
            "IDAQuadInit",
            file!(),
            MSG_MEM_FAIL,
        );
        return IDA_MEM_FAIL;
    }

    /* Initialize phiQ in the history array */
    let phiQ0 = IDA_mem.borrow().ida_phiQ[0].clone().unwrap();
    N_VScale(ONE, yQ0, &phiQ0);

    /* C: N_VConstVectorArray(maxord, ZERO, IDA_mem->ida_phiQ + 1) — the
    tail slice phiQ[1 .. maxord] of the (maxord+1)-long allocated block */
    let (maxord, phiQ_tail) = {
        let m = IDA_mem.borrow();
        let maxord = m.ida_maxord;
        let tail: Vec<N_Vector> = (1..=maxord as usize)
            .map(|j| m.ida_phiQ[j].clone().unwrap())
            .collect();
        (maxord, tail)
    };
    let retval = N_VConstVectorArray(maxord, ZERO, &phiQ_tail);
    if retval != IDA_SUCCESS {
        return IDA_VECTOROP_ERR;
    }

    {
        let mut m = IDA_mem.borrow_mut();

        /* Copy the input parameters into IDAS state */
        m.ida_rhsQ = Some(rhsQ);

        /* Initialize counters */
        m.ida_nrQe = 0;
        m.ida_netfQ = 0;

        /* Quadrature integration turned ON */
        m.ida_quadr = SUNTRUE;
        m.ida_quadMallocDone = SUNTRUE;
    }

    /* Quadrature initialization was successful */
    IDA_SUCCESS
}

/*-----------------------------------------------------------------*/

/*
 * IDAQuadReInit
 *
 * IDAQuadReInit re-initializes IDAS's quadrature related memory
 * for a problem, assuming it has already been allocated in prior
 * calls to IDAInit and IDAQuadMalloc.
 * All problem specification inputs are checked for errors.
 * If any error occurs during initialization, it is reported to the
 * file whose file pointer is errfp.
 * The return value is IDA_SUCCESS = 0 if no errors occurred, or
 * a negative value otherwise.
 */

pub fn IDAQuadReInit(ida_mem: &IDAMem, yQ0: &N_Vector) -> i32 {
    /* Check ida_mem */
    /* NULL-mem check: handled by type system */
    let IDA_mem = ida_mem;

    /* Check if quadrature was initialized */
    if !IDA_mem.borrow().ida_quadMallocDone {
        IDAProcessError(
            Some(IDA_mem),
            IDA_NO_QUAD,
            line!() as i32,
            "IDAQuadReInit",
            file!(),
            MSG_NO_QUAD,
        );
        return IDA_NO_QUAD;
    }

    /* Initialize phiQ in the history array */
    let phiQ0 = IDA_mem.borrow().ida_phiQ[0].clone().unwrap();
    N_VScale(ONE, yQ0, &phiQ0);

    let (maxord, phiQ_tail) = {
        let m = IDA_mem.borrow();
        let maxord = m.ida_maxord;
        let tail: Vec<N_Vector> = (1..=maxord as usize)
            .map(|j| m.ida_phiQ[j].clone().unwrap())
            .collect();
        (maxord, tail)
    };
    let retval = N_VConstVectorArray(maxord, ZERO, &phiQ_tail);
    if retval != IDA_SUCCESS {
        return IDA_VECTOROP_ERR;
    }

    {
        let mut m = IDA_mem.borrow_mut();

        /* Initialize counters */
        m.ida_nrQe = 0;
        m.ida_netfQ = 0;

        /* Quadrature integration turned ON */
        m.ida_quadr = SUNTRUE;
    }

    /* Quadrature re-initialization was successful */
    IDA_SUCCESS
}

/*
 * IDAQuadSStolerances
 * IDAQuadSVtolerances
 *
 *
 * These functions specify the integration tolerances for quadrature
 * variables. One of them MUST be called before the first call to
 * IDA IF error control on the quadrature variables is enabled
 * (see IDASetQuadErrCon).
 *
 * IDASStolerances specifies scalar relative and absolute tolerances.
 * IDASVtolerances specifies scalar relative tolerance and a vector
 *   absolute tolerance (a potentially different absolute tolerance
 *   for each vector component).
 */
pub fn IDAQuadSStolerances(ida_mem: &IDAMem, reltolQ: sunrealtype, abstolQ: sunrealtype) -> i32 {
    /* Check ida mem: NULL-mem check handled by type system */
    let IDA_mem = ida_mem;

    /* Check if quadrature was initialized */
    if !IDA_mem.borrow().ida_quadMallocDone {
        IDAProcessError(
            Some(IDA_mem),
            IDA_NO_QUAD,
            line!() as i32,
            "IDAQuadSStolerances",
            file!(),
            MSG_NO_QUAD,
        );
        return IDA_NO_QUAD;
    }

    /* Test user-supplied tolerances */
    if reltolQ < ZERO {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDAQuadSStolerances",
            file!(),
            MSG_BAD_RTOLQ,
        );
        return IDA_ILL_INPUT;
    }

    if abstolQ < ZERO {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDAQuadSStolerances",
            file!(),
            MSG_BAD_ATOLQ,
        );
        return IDA_ILL_INPUT;
    }

    /* Copy tolerances into memory */
    let mut m = IDA_mem.borrow_mut();
    m.ida_itolQ = IDA_SS;

    m.ida_rtolQ = reltolQ;
    m.ida_SatolQ = abstolQ;
    m.ida_atolQmin0 = abstolQ == ZERO;

    IDA_SUCCESS
}

pub fn IDAQuadSVtolerances(ida_mem: &IDAMem, reltolQ: sunrealtype, abstolQ: &N_Vector) -> i32 {
    /* Check ida mem: NULL-mem check handled by type system */
    let IDA_mem = ida_mem;

    /* Check if quadrature was initialized */
    if !IDA_mem.borrow().ida_quadMallocDone {
        IDAProcessError(
            Some(IDA_mem),
            IDA_NO_QUAD,
            line!() as i32,
            "IDAQuadSVtolerances",
            file!(),
            MSG_NO_QUAD,
        );
        return IDA_NO_QUAD;
    }

    /* Test user-supplied tolerances */
    if reltolQ < ZERO {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDAQuadSVtolerances",
            file!(),
            MSG_BAD_RTOLQ,
        );
        return IDA_ILL_INPUT;
    }

    /* NULL abstolQ check: handled by type system */

    let atolmin = N_VMin(abstolQ);
    if atolmin < ZERO {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDAQuadSVtolerances",
            file!(),
            MSG_BAD_ATOLQ,
        );
        return IDA_ILL_INPUT;
    }

    /* Copy tolerances into memory */
    {
        let mut m = IDA_mem.borrow_mut();
        m.ida_itolQ = IDA_SV;
        m.ida_rtolQ = reltolQ;
    }

    /* clone the absolute tolerances vector (if necessary) */
    if !IDA_mem.borrow().ida_VatolQMallocDone {
        let vatolQ = N_VClone(abstolQ).unwrap();
        let mut m = IDA_mem.borrow_mut();
        m.ida_VatolQ = Some(vatolQ);
        m.ida_lrw += m.ida_lrw1Q;
        m.ida_liw += m.ida_liw1Q;
        m.ida_VatolQMallocDone = SUNTRUE;
    }

    let vatolQ = IDA_mem.borrow().ida_VatolQ.clone().unwrap();
    N_VScale(ONE, abstolQ, &vatolQ);
    IDA_mem.borrow_mut().ida_atolQmin0 = atolmin == ZERO;

    IDA_SUCCESS
}

/*
 * IDASenMalloc
 *
 * IDASensInit allocates and initializes sensitivity related
 * memory for a problem. All problem specification inputs are
 * checked for errors. If any error occurs during initialization,
 * it is reported to the file whose file pointer is errfp.
 * The return value is IDA_SUCCESS = 0 if no errors occurred, or
 * a negative value otherwise.
 */

pub fn IDASensInit(
    ida_mem: &IDAMem,
    Ns: i32,
    ism: i32,
    fS: Option<IDASensResFn>,
    yS0: &[N_Vector],
    ypS0: &[N_Vector],
) -> i32 {
    /* Check ida_mem */
    /* NULL-mem check: handled by type system */
    let IDA_mem = ida_mem;

    /* Check if Ns is legal */
    if Ns <= 0 {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDASensInit",
            file!(),
            MSG_BAD_NS,
        );
        return IDA_ILL_INPUT;
    }
    IDA_mem.borrow_mut().ida_Ns = Ns;

    /* Check if ism is legal */
    if (ism != IDA_SIMULTANEOUS) && (ism != IDA_STAGGERED) {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDASensInit",
            file!(),
            MSG_BAD_ISM,
        );
        return IDA_ILL_INPUT;
    }
    IDA_mem.borrow_mut().ida_ism = ism;

    /* Check if yS0 and ypS0 are non-null: handled by type system */

    /* Store sensitivity RHS-related data */

    match fS {
        Some(fS) => {
            let mut m = IDA_mem.borrow_mut();
            m.ida_resS = Some(fS);
            /* C: ida_user_dataS = ida_user_data (pointer alias); `None`
            means "pass the integrator's ida_user_data" at call time */
            m.ida_user_dataS = None;
            m.ida_resSDQ = SUNFALSE;
        }
        None => {
            let token: Box<dyn Any> = Box::new(IDA_mem.clone());
            let mut m = IDA_mem.borrow_mut();
            m.ida_resS = Some(IDASensResDQ);
            m.ida_user_dataS = Some(token);
            m.ida_resSDQ = SUNTRUE;
        }
    }

    /* Allocate the vectors (using yS0[0] as a template) */

    let allocOK = IDASensAllocVectors(IDA_mem, &yS0[0]);
    if !allocOK {
        IDAProcessError(
            Some(IDA_mem),
            IDA_MEM_FAIL,
            line!() as i32,
            "IDASensInit",
            file!(),
            MSG_MEM_FAIL,
        );
        return IDA_MEM_FAIL;
    }

    /* Allocate temporary work arrays for fused vector ops (the C
    NULL-check failure branch is unreachable: Vec allocation aborts
    rather than returning NULL) */
    if Ns as usize * MXORDP1 > MXORDP1 {
        let mut m = IDA_mem.borrow_mut();
        m.ida_cvals = vec![ZERO; Ns as usize * MXORDP1];
        m.ida_Xvecs = Vec::new();
        m.ida_Zvecs = Vec::new();
    }

    /*----------------------------------------------
    All error checking is complete at this point
    -----------------------------------------------*/

    /* Initialize the phiS array */
    let (cvals, phiS0, phiS1) = {
        let mut m = IDA_mem.borrow_mut();
        for is in 0..Ns as usize {
            m.ida_cvals[is] = ONE;
        }
        let cvals = m.ida_cvals.clone();
        let phiS0 = m.ida_phiS[0].clone();
        let phiS1 = m.ida_phiS[1].clone();
        (cvals, phiS0, phiS1)
    };

    let retval = N_VScaleVectorArray(Ns, &cvals, yS0, &phiS0);
    if retval != IDA_SUCCESS {
        return IDA_VECTOROP_ERR;
    }

    let retval = N_VScaleVectorArray(Ns, &cvals, ypS0, &phiS1);
    if retval != IDA_SUCCESS {
        return IDA_VECTOROP_ERR;
    }

    {
        let mut m = IDA_mem.borrow_mut();

        /* Initialize all sensitivity related counters */
        m.ida_nrSe = 0;
        m.ida_nreS = 0;
        m.ida_ncfnS = 0;
        m.ida_netfS = 0;
        m.ida_nniS = 0;
        m.ida_nnfS = 0;
        m.ida_nsetupsS = 0;

        /* Set default values for plist and pbar */
        for is in 0..Ns as usize {
            m.ida_plist[is] = is as i32;
            m.ida_pbar[is] = ONE;
        }

        /* Sensitivities will be computed */
        m.ida_sensi = SUNTRUE;
        m.ida_sensMallocDone = SUNTRUE;
    }

    /* create a Newton nonlinear solver object by default */
    let (delta, sunctx) = {
        let m = IDA_mem.borrow();
        (m.ida_delta.clone().unwrap(), m.ida_sunctx.clone())
    };
    let NLS = if ism == IDA_SIMULTANEOUS {
        SUNNonlinSol_NewtonSens(Ns + 1, &delta, &sunctx)
    } else {
        SUNNonlinSol_NewtonSens(Ns, &delta, &sunctx)
    };

    /* check that the nonlinear solver is non-NULL */
    let NLS = match NLS {
        Some(nls) => nls,
        None => {
            IDAProcessError(
                Some(IDA_mem),
                IDA_MEM_FAIL,
                line!() as i32,
                "IDASensInit",
                file!(),
                MSG_MEM_FAIL,
            );
            IDASensFreeVectors(IDA_mem);
            return IDA_MEM_FAIL;
        }
    };

    /* attach the nonlinear solver to the IDA memory */
    let retval = if ism == IDA_SIMULTANEOUS {
        crate::idas_nls_sim::IDASetNonlinearSolverSensSim(IDA_mem, &NLS)
    } else {
        crate::idas_nls_stg::IDASetNonlinearSolverSensStg(IDA_mem, &NLS)
    };

    /* check that the nonlinear solver was successfully attached */
    if retval != IDA_SUCCESS {
        IDAProcessError(
            Some(IDA_mem),
            retval,
            line!() as i32,
            "IDASensInit",
            file!(),
            "Setting the nonlinear solver failed",
        );
        IDASensFreeVectors(IDA_mem);
        let _ = SUNNonlinSolFree(Some(NLS));
        return IDA_MEM_FAIL;
    }

    /* set ownership flag */
    if ism == IDA_SIMULTANEOUS {
        IDA_mem.borrow_mut().ownNLSsim = SUNTRUE;
    } else {
        IDA_mem.borrow_mut().ownNLSstg = SUNTRUE;
    }

    /* Sensitivity initialization was successful */
    IDA_SUCCESS
}

/*-----------------------------------------------------------------*/

/*
 * IDASensReInit
 *
 * IDASensReInit re-initializes IDAS's sensitivity related memory
 * for a problem, assuming it has already been allocated in prior
 * calls to IDAInit and IDASensInit.
 * All problem specification inputs are checked for errors.
 * The number of sensitivities Ns is assumed to be unchanged since
 * the previous call to IDASensInit.
 * If any error occurs during initialization, it is reported to the
 * file whose file pointer is errfp.
 * The return value is IDA_SUCCESS = 0 if no errors occurred, or
 * a negative value otherwise.
 */

pub fn IDASensReInit(ida_mem: &IDAMem, ism: i32, yS0: &[N_Vector], ypS0: &[N_Vector]) -> i32 {
    /* Check ida_mem */
    /* NULL-mem check: handled by type system */
    let IDA_mem = ida_mem;

    /* Was sensitivity initialized? */
    if !IDA_mem.borrow().ida_sensMallocDone {
        IDAProcessError(
            Some(IDA_mem),
            IDA_NO_SENS,
            line!() as i32,
            "IDASensReInit",
            file!(),
            MSG_NO_SENSI,
        );
        return IDA_NO_SENS;
    }

    /* Check if ism is legal */
    if (ism != IDA_SIMULTANEOUS) && (ism != IDA_STAGGERED) {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDASensReInit",
            file!(),
            MSG_BAD_ISM,
        );
        return IDA_ILL_INPUT;
    }
    IDA_mem.borrow_mut().ida_ism = ism;

    /* Check if yS0 and ypS0 are non-null: handled by type system */

    /*-----------------------------------------------
    All error checking is complete at this point
    -----------------------------------------------*/

    /* Initialize the phiS array */
    let (Ns, cvals, phiS0, phiS1) = {
        let mut m = IDA_mem.borrow_mut();
        let Ns = m.ida_Ns;
        for is in 0..Ns as usize {
            m.ida_cvals[is] = ONE;
        }
        let cvals = m.ida_cvals.clone();
        let phiS0 = m.ida_phiS[0].clone();
        let phiS1 = m.ida_phiS[1].clone();
        (Ns, cvals, phiS0, phiS1)
    };

    let retval = N_VScaleVectorArray(Ns, &cvals, yS0, &phiS0);
    if retval != IDA_SUCCESS {
        return IDA_VECTOROP_ERR;
    }

    let retval = N_VScaleVectorArray(Ns, &cvals, ypS0, &phiS1);
    if retval != IDA_SUCCESS {
        return IDA_VECTOROP_ERR;
    }

    {
        let mut m = IDA_mem.borrow_mut();

        /* Initialize all sensitivity related counters */
        m.ida_nrSe = 0;
        m.ida_nreS = 0;
        m.ida_ncfnS = 0;
        m.ida_netfS = 0;
        m.ida_nniS = 0;
        m.ida_nnfS = 0;
        m.ida_nsetupsS = 0;

        /* Set default values for plist and pbar */
        for is in 0..Ns as usize {
            m.ida_plist[is] = is as i32;
            m.ida_pbar[is] = ONE;
        }

        /* Sensitivities will be computed */
        m.ida_sensi = SUNTRUE;
    }

    /* Check if the NLS exists, create the default NLS if needed */
    let need_nls = {
        let m = IDA_mem.borrow();
        (ism == IDA_SIMULTANEOUS && m.NLSsim.is_none())
            || (ism == IDA_STAGGERED && m.NLSstg.is_none())
    };
    if need_nls {
        /* create a Newton nonlinear solver object by default */
        let (delta, sunctx) = {
            let m = IDA_mem.borrow();
            (m.ida_delta.clone().unwrap(), m.ida_sunctx.clone())
        };
        let NLS = if ism == IDA_SIMULTANEOUS {
            SUNNonlinSol_NewtonSens(Ns + 1, &delta, &sunctx)
        } else {
            SUNNonlinSol_NewtonSens(Ns, &delta, &sunctx)
        };

        /* check that the nonlinear solver is non-NULL */
        let NLS = match NLS {
            Some(nls) => nls,
            None => {
                IDAProcessError(
                    Some(IDA_mem),
                    IDA_MEM_FAIL,
                    line!() as i32,
                    "IDASensReInit",
                    file!(),
                    MSG_MEM_FAIL,
                );
                return IDA_MEM_FAIL;
            }
        };

        /* attach the nonlinear solver to the IDA memory */
        let retval = if ism == IDA_SIMULTANEOUS {
            crate::idas_nls_sim::IDASetNonlinearSolverSensSim(IDA_mem, &NLS)
        } else {
            crate::idas_nls_stg::IDASetNonlinearSolverSensStg(IDA_mem, &NLS)
        };

        /* check that the nonlinear solver was successfully attached */
        if retval != IDA_SUCCESS {
            IDAProcessError(
                Some(IDA_mem),
                retval,
                line!() as i32,
                "IDASensReInit",
                file!(),
                "Setting the nonlinear solver failed",
            );
            let _ = SUNNonlinSolFree(Some(NLS));
            return IDA_MEM_FAIL;
        }

        /* set ownership flag */
        if ism == IDA_SIMULTANEOUS {
            IDA_mem.borrow_mut().ownNLSsim = SUNTRUE;
        } else {
            IDA_mem.borrow_mut().ownNLSstg = SUNTRUE;
        }

        /* initialize the NLS object, this assumes that the linear solver has
        already been initialized in IDAInit */
        let retval = if ism == IDA_SIMULTANEOUS {
            crate::idas_nls_sim::idaNlsInitSensSim(IDA_mem)
        } else {
            crate::idas_nls_stg::idaNlsInitSensStg(IDA_mem)
        };

        if retval != IDA_SUCCESS {
            IDAProcessError(
                Some(IDA_mem),
                IDA_NLS_INIT_FAIL,
                line!() as i32,
                "IDASensReInit",
                file!(),
                MSG_NLS_INIT_FAIL,
            );
            return IDA_NLS_INIT_FAIL;
        }
    }

    /* Sensitivity re-initialization was successful */
    IDA_SUCCESS
}

/*-----------------------------------------------------------------*/

/*
 * IDASensSStolerances
 * IDASensSVtolerances
 * IDASensEEtolerances
 *
 * These functions specify the integration tolerances for sensitivity
 * variables. One of them MUST be called before the first call to IDASolve.
 *
 * IDASensSStolerances specifies scalar relative and absolute tolerances.
 * IDASensSVtolerances specifies scalar relative tolerance and a vector
 *   absolute tolerance for each sensitivity vector (a potentially different
 *   absolute tolerance for each vector component).
 * IDASensEEtolerances specifies that tolerances for sensitivity variables
 *   should be estimated from those provided for the state variables.
 */

pub fn IDASensSStolerances(ida_mem: &IDAMem, reltolS: sunrealtype, abstolS: &[sunrealtype]) -> i32 {
    /* Check ida_mem pointer: NULL-mem check handled by type system */
    let IDA_mem = ida_mem;

    /* Was sensitivity initialized? */

    if !IDA_mem.borrow().ida_sensMallocDone {
        IDAProcessError(
            Some(IDA_mem),
            IDA_NO_SENS,
            line!() as i32,
            "IDASensSStolerances",
            file!(),
            MSG_NO_SENSI,
        );
        return IDA_NO_SENS;
    }

    /* Test user-supplied tolerances */

    if reltolS < ZERO {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDASensSStolerances",
            file!(),
            MSG_BAD_RTOLS,
        );
        return IDA_ILL_INPUT;
    }

    /* NULL abstolS check: handled by type system */

    let Ns = IDA_mem.borrow().ida_Ns;
    for is in 0..Ns as usize {
        if abstolS[is] < ZERO {
            IDAProcessError(
                Some(IDA_mem),
                IDA_ILL_INPUT,
                line!() as i32,
                "IDASensSStolerances",
                file!(),
                MSG_BAD_ATOLS,
            );
            return IDA_ILL_INPUT;
        }
    }

    /* Copy tolerances into memory */

    let mut m = IDA_mem.borrow_mut();
    m.ida_itolS = IDA_SS;

    m.ida_rtolS = reltolS;

    if !m.ida_SatolSMallocDone {
        m.ida_SatolS = vec![ZERO; Ns as usize];
        m.ida_atolSmin0 = vec![SUNFALSE; Ns as usize];
        m.ida_lrw += Ns as i64;
        m.ida_SatolSMallocDone = SUNTRUE;
    }

    for is in 0..Ns as usize {
        m.ida_SatolS[is] = abstolS[is];
        m.ida_atolSmin0[is] = abstolS[is] == ZERO;
    }

    IDA_SUCCESS
}

pub fn IDASensSVtolerances(ida_mem: &IDAMem, reltolS: sunrealtype, abstolS: &[N_Vector]) -> i32 {
    /* Check ida_mem pointer: NULL-mem check handled by type system */
    let IDA_mem = ida_mem;

    /* Was sensitivity initialized? */

    if !IDA_mem.borrow().ida_sensMallocDone {
        IDAProcessError(
            Some(IDA_mem),
            IDA_NO_SENS,
            line!() as i32,
            "IDASensSVtolerances",
            file!(),
            MSG_NO_SENSI,
        );
        return IDA_NO_SENS;
    }

    /* Test user-supplied tolerances */

    if reltolS < ZERO {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDASensSVtolerances",
            file!(),
            MSG_BAD_RTOLS,
        );
        return IDA_ILL_INPUT;
    }

    /* NULL abstolS check: handled by type system */

    let Ns = IDA_mem.borrow().ida_Ns;
    let mut atolmin: Vec<sunrealtype> = vec![ZERO; Ns as usize];
    for is in 0..Ns as usize {
        atolmin[is] = N_VMin(&abstolS[is]);
        if atolmin[is] < ZERO {
            IDAProcessError(
                Some(IDA_mem),
                IDA_ILL_INPUT,
                line!() as i32,
                "IDASensSVtolerances",
                file!(),
                MSG_BAD_ATOLS,
            );
            /* C: free(atolmin) — the Vec is dropped on return */
            return IDA_ILL_INPUT;
        }
    }

    {
        let mut m = IDA_mem.borrow_mut();
        m.ida_itolS = IDA_SV;
        m.ida_rtolS = reltolS;
    }

    if !IDA_mem.borrow().ida_VatolSMallocDone {
        let tempv1 = IDA_mem.borrow().ida_tempv1.clone().unwrap();
        let vatolS = N_VCloneVectorArray(Ns, &tempv1).unwrap();
        let mut m = IDA_mem.borrow_mut();
        m.ida_VatolS = vatolS;
        m.ida_atolSmin0 = vec![SUNFALSE; Ns as usize];
        m.ida_lrw += Ns as i64 * m.ida_lrw1;
        m.ida_liw += Ns as i64 * m.ida_liw1;
        m.ida_VatolSMallocDone = SUNTRUE;
    }

    let (cvals, vatolS) = {
        let mut m = IDA_mem.borrow_mut();
        for is in 0..Ns as usize {
            m.ida_cvals[is] = ONE;
            m.ida_atolSmin0[is] = atolmin[is] == ZERO;
        }
        let cvals = m.ida_cvals.clone();
        let vatolS = m.ida_VatolS.clone();
        (cvals, vatolS)
    };
    drop(atolmin);

    let retval = N_VScaleVectorArray(Ns, &cvals, abstolS, &vatolS);
    if retval != IDA_SUCCESS {
        return IDA_VECTOROP_ERR;
    }

    IDA_SUCCESS
}

pub fn IDASensEEtolerances(ida_mem: &IDAMem) -> i32 {
    /* NULL-mem check: handled by type system */
    let IDA_mem = ida_mem;

    /* Was sensitivity initialized? */

    if !IDA_mem.borrow().ida_sensMallocDone {
        IDAProcessError(
            Some(IDA_mem),
            IDA_NO_SENS,
            line!() as i32,
            "IDASensEEtolerances",
            file!(),
            MSG_NO_SENSI,
        );
        return IDA_NO_SENS;
    }

    IDA_mem.borrow_mut().ida_itolS = IDA_EE;

    IDA_SUCCESS
}

pub fn IDAQuadSensInit(
    ida_mem: &IDAMem,
    rhsQS: Option<IDAQuadSensRhsFn>,
    yQS0: &[N_Vector],
) -> i32 {
    /* NULL-mem check: handled by type system */
    let IDA_mem = ida_mem;

    /* Check if sensitivity analysis is active */
    if !IDA_mem.borrow().ida_sensi {
        /* NOTE: upstream passes a NULL mem here, so this message goes to the
        global fallback error handler — preserved verbatim */
        IDAProcessError(
            None,
            IDA_NO_SENS,
            line!() as i32,
            "IDAQuadSensInit",
            file!(),
            MSG_NO_SENSI,
        );
        return IDA_NO_SENS;
    }

    /* Verify yQS0 parameter: NULL check handled by type system */

    /* Allocate vector needed for quadratures' sensitivities. */
    let allocOK = IDAQuadSensAllocVectors(IDA_mem, &yQS0[0]);
    if !allocOK {
        /* NOTE: upstream passes a NULL mem here — preserved verbatim */
        IDAProcessError(
            None,
            IDA_MEM_FAIL,
            line!() as i32,
            "IDAQuadSensInit",
            file!(),
            MSG_MEM_FAIL,
        );
        return IDA_MEM_FAIL;
    }

    /* Error checking complete. */
    match rhsQS {
        None => {
            let token: Box<dyn Any> = Box::new(IDA_mem.clone());
            let mut m = IDA_mem.borrow_mut();
            m.ida_rhsQSDQ = SUNTRUE;
            m.ida_rhsQS = Some(IDAQuadSensRhsInternalDQ);

            m.ida_user_dataQS = Some(token);
        }
        Some(rhsQS) => {
            let mut m = IDA_mem.borrow_mut();
            m.ida_rhsQSDQ = SUNFALSE;
            m.ida_rhsQS = Some(rhsQS);

            /* C: ida_user_dataQS = ida_user_data (pointer alias); `None`
            means "pass the integrator's ida_user_data" at call time */
            m.ida_user_dataQS = None;
        }
    }

    /* Initialize phiQS[0] in the history array */
    let (Ns, cvals, phiQS0) = {
        let mut m = IDA_mem.borrow_mut();
        let Ns = m.ida_Ns;
        for is in 0..Ns as usize {
            m.ida_cvals[is] = ONE;
        }
        let cvals = m.ida_cvals.clone();
        let phiQS0 = m.ida_phiQS[0].clone();
        (Ns, cvals, phiQS0)
    };

    let retval = N_VScaleVectorArray(Ns, &cvals, yQS0, &phiQS0);
    if retval != IDA_SUCCESS {
        return IDA_VECTOROP_ERR;
    }

    {
        let mut m = IDA_mem.borrow_mut();

        /* Initialize all sensitivities related counters. */
        m.ida_nrQSe = 0;
        m.ida_nrQeS = 0;
        m.ida_netfQS = 0;

        /* Everything all right, set the flags and return with success. */
        m.ida_quadr_sensi = SUNTRUE;
        m.ida_quadSensMallocDone = SUNTRUE;
    }

    IDA_SUCCESS
}

pub fn IDAQuadSensReInit(ida_mem: &IDAMem, yQS0: &[N_Vector]) -> i32 {
    /* NULL-mem check: handled by type system */
    let IDA_mem = ida_mem;

    /* Check if sensitivity analysis is active */
    if !IDA_mem.borrow().ida_sensi {
        IDAProcessError(
            Some(IDA_mem),
            IDA_NO_SENS,
            line!() as i32,
            "IDAQuadSensReInit",
            file!(),
            MSG_NO_SENSI,
        );
        return IDA_NO_SENS;
    }

    /* Was sensitivity for quadrature already initialized? */
    if !IDA_mem.borrow().ida_quadSensMallocDone {
        IDAProcessError(
            Some(IDA_mem),
            IDA_NO_QUADSENS,
            line!() as i32,
            "IDAQuadSensReInit",
            file!(),
            MSG_NO_QUADSENSI,
        );
        return IDA_NO_QUADSENS;
    }

    /* Verify yQS0 parameter: NULL check handled by type system (upstream
    reports MSG_NULL_YQS0 through the global fallback handler) */

    /* Error checking complete at this point. */

    /* Initialize phiQS[0] in the history array */
    let (Ns, cvals, phiQS0) = {
        let mut m = IDA_mem.borrow_mut();
        let Ns = m.ida_Ns;
        for is in 0..Ns as usize {
            m.ida_cvals[is] = ONE;
        }
        let cvals = m.ida_cvals.clone();
        let phiQS0 = m.ida_phiQS[0].clone();
        (Ns, cvals, phiQS0)
    };

    let retval = N_VScaleVectorArray(Ns, &cvals, yQS0, &phiQS0);
    if retval != IDA_SUCCESS {
        return IDA_VECTOROP_ERR;
    }

    {
        let mut m = IDA_mem.borrow_mut();

        /* Initialize all sensitivities related counters. */
        m.ida_nrQSe = 0;
        m.ida_nrQeS = 0;
        m.ida_netfQS = 0;

        /* Everything all right, set the flags and return with success. */
        m.ida_quadr_sensi = SUNTRUE;
    }

    IDA_SUCCESS
}

/*
 * IDAQuadSensSStolerances
 * IDAQuadSensSVtolerances
 * IDAQuadSensEEtolerances
 *
 * These functions specify the integration tolerances for quadrature
 * sensitivity variables. One of them MUST be called before the first
 * call to IDAS IF these variables are included in the error test.
 *
 * IDAQuadSensSStolerances specifies scalar relative and absolute tolerances.
 * IDAQuadSensSVtolerances specifies scalar relative tolerance and a vector
 *   absolute tolerance for each quadrature sensitivity vector (a potentially
 *   different absolute tolerance for each vector component).
 * IDAQuadSensEEtolerances specifies that tolerances for sensitivity variables
 *   should be estimated from those provided for the quadrature variables.
 *   In this case, tolerances for the quadrature variables must be
 *   specified through a call to one of IDAQuad**tolerances.
 */

pub fn IDAQuadSensSStolerances(
    ida_mem: &IDAMem,
    reltolQS: sunrealtype,
    abstolQS: &[sunrealtype],
) -> i32 {
    /* NULL-mem check: handled by type system */
    let IDA_mem = ida_mem;

    /* Check if sensitivity analysis is active */
    if !IDA_mem.borrow().ida_sensi {
        IDAProcessError(
            Some(IDA_mem),
            IDA_NO_SENS,
            line!() as i32,
            "IDAQuadSensSStolerances",
            file!(),
            MSG_NO_SENSI,
        );
        return IDA_NO_SENS;
    }

    /* Was sensitivity for quadrature already initialized? */
    if !IDA_mem.borrow().ida_quadSensMallocDone {
        IDAProcessError(
            Some(IDA_mem),
            IDA_NO_QUADSENS,
            line!() as i32,
            "IDAQuadSensSStolerances",
            file!(),
            MSG_NO_QUADSENSI,
        );
        return IDA_NO_QUADSENS;
    }

    /* Test user-supplied tolerances */

    if reltolQS < ZERO {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDAQuadSensSStolerances",
            file!(),
            MSG_BAD_RELTOLQS,
        );
        return IDA_ILL_INPUT;
    }

    /* NULL abstolQS check: handled by type system */

    let Ns = IDA_mem.borrow().ida_Ns;
    for is in 0..Ns as usize {
        if abstolQS[is] < ZERO {
            IDAProcessError(
                Some(IDA_mem),
                IDA_ILL_INPUT,
                line!() as i32,
                "IDAQuadSensSStolerances",
                file!(),
                MSG_BAD_ABSTOLQS,
            );
            return IDA_ILL_INPUT;
        }
    }

    /* Save data. */
    let mut m = IDA_mem.borrow_mut();
    m.ida_itolQS = IDA_SS;
    m.ida_rtolQS = reltolQS;

    if !m.ida_SatolQSMallocDone {
        m.ida_SatolQS = vec![ZERO; Ns as usize];
        m.ida_atolQSmin0 = vec![SUNFALSE; Ns as usize];
        m.ida_lrw += Ns as i64;
        m.ida_SatolQSMallocDone = SUNTRUE;
    }

    for is in 0..Ns as usize {
        m.ida_SatolQS[is] = abstolQS[is];
        m.ida_atolQSmin0[is] = abstolQS[is] == ZERO;
    }

    IDA_SUCCESS
}

pub fn IDAQuadSensSVtolerances(
    ida_mem: &IDAMem,
    reltolQS: sunrealtype,
    abstolQS: &[N_Vector],
) -> i32 {
    /* NULL-mem check: handled by type system */
    let IDA_mem = ida_mem;

    /* Check if sensitivity analysis is active */
    if !IDA_mem.borrow().ida_sensi {
        IDAProcessError(
            Some(IDA_mem),
            IDA_NO_SENS,
            line!() as i32,
            "IDAQuadSensSVtolerances",
            file!(),
            MSG_NO_SENSI,
        );
        return IDA_NO_SENS;
    }

    /* Was sensitivity for quadrature already initialized? */
    if !IDA_mem.borrow().ida_quadSensMallocDone {
        IDAProcessError(
            Some(IDA_mem),
            IDA_NO_QUADSENS,
            line!() as i32,
            "IDAQuadSensSVtolerances",
            file!(),
            MSG_NO_QUADSENSI,
        );
        return IDA_NO_QUADSENS;
    }

    /* Test user-supplied tolerances */

    if reltolQS < ZERO {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDAQuadSensSVtolerances",
            file!(),
            MSG_BAD_RELTOLQS,
        );
        return IDA_ILL_INPUT;
    }

    /* NULL abstolQS check: handled by type system */

    let Ns = IDA_mem.borrow().ida_Ns;
    let mut atolmin: Vec<sunrealtype> = vec![ZERO; Ns as usize];
    for is in 0..Ns as usize {
        atolmin[is] = N_VMin(&abstolQS[is]);
        if atolmin[is] < ZERO {
            IDAProcessError(
                Some(IDA_mem),
                IDA_ILL_INPUT,
                line!() as i32,
                "IDAQuadSensSVtolerances",
                file!(),
                MSG_BAD_ABSTOLQS,
            );
            /* C: free(atolmin) — the Vec is dropped on return */
            return IDA_ILL_INPUT;
        }
    }

    /* Save data. */
    {
        let mut m = IDA_mem.borrow_mut();
        m.ida_itolQS = IDA_SV;
        m.ida_rtolQS = reltolQS;
    }

    if !IDA_mem.borrow().ida_VatolQSMallocDone {
        let vatolQS = N_VCloneVectorArray(Ns, &abstolQS[0]).unwrap();
        let mut m = IDA_mem.borrow_mut();
        m.ida_VatolQS = vatolQS;
        m.ida_atolQSmin0 = vec![SUNFALSE; Ns as usize];
        m.ida_lrw += Ns as i64 * m.ida_lrw1Q;
        m.ida_liw += Ns as i64 * m.ida_liw1Q;
        m.ida_VatolQSMallocDone = SUNTRUE;
    }

    let (cvals, vatolQS) = {
        let mut m = IDA_mem.borrow_mut();
        for is in 0..Ns as usize {
            m.ida_cvals[is] = ONE;
            m.ida_atolQSmin0[is] = atolmin[is] == ZERO;
        }
        let cvals = m.ida_cvals.clone();
        let vatolQS = m.ida_VatolQS.clone();
        (cvals, vatolQS)
    };
    drop(atolmin);

    let retval = N_VScaleVectorArray(Ns, &cvals, abstolQS, &vatolQS);
    if retval != IDA_SUCCESS {
        return IDA_VECTOROP_ERR;
    }

    IDA_SUCCESS
}

pub fn IDAQuadSensEEtolerances(ida_mem: &IDAMem) -> i32 {
    /* NULL-mem check: handled by type system */
    let IDA_mem = ida_mem;

    /* Check if sensitivity analysis is active */
    if !IDA_mem.borrow().ida_sensi {
        IDAProcessError(
            Some(IDA_mem),
            IDA_NO_SENS,
            line!() as i32,
            "IDAQuadSensEEtolerances",
            file!(),
            MSG_NO_SENSI,
        );
        return IDA_NO_SENS;
    }

    /* Was sensitivity for quadrature already initialized? */
    if !IDA_mem.borrow().ida_quadSensMallocDone {
        IDAProcessError(
            Some(IDA_mem),
            IDA_NO_QUADSENS,
            line!() as i32,
            "IDAQuadSensEEtolerances",
            file!(),
            MSG_NO_QUADSENSI,
        );
        return IDA_NO_QUADSENS;
    }

    IDA_mem.borrow_mut().ida_itolQS = IDA_EE;

    IDA_SUCCESS
}

/*
 * IDASensToggleOff
 *
 * IDASensToggleOff deactivates sensitivity calculations.
 * It does NOT deallocate sensitivity-related memory.
 */
pub fn IDASensToggleOff(ida_mem: &IDAMem) -> i32 {
    /* Check ida_mem: NULL-mem check handled by type system */
    let IDA_mem = ida_mem;

    /* Disable sensitivities */
    let mut m = IDA_mem.borrow_mut();
    m.ida_sensi = SUNFALSE;
    m.ida_quadr_sensi = SUNFALSE;

    IDA_SUCCESS
}

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

    /* Sensitivity-specific tests (if using internal DQ functions) */
    let (sensi, resSDQ) = {
        let m = IDA_mem.borrow();
        (m.ida_sensi, m.ida_resSDQ)
    };
    if sensi && resSDQ {
        /* Make sure we have the right 'user data' */
        let token: Box<dyn Any> = Box::new(IDA_mem.clone());
        IDA_mem.borrow_mut().ida_user_dataS = Some(token);
        /* Test if we have the problem parameters */
        if IDA_mem.borrow().ida_p.is_none() {
            IDAProcessError(
                Some(IDA_mem),
                IDA_ILL_INPUT,
                line!() as i32,
                "IDASolve",
                file!(),
                MSG_NULL_P,
            );
            return IDA_ILL_INPUT;
        }
    }

    let (quadr_sensi, rhsQSDQ) = {
        let m = IDA_mem.borrow();
        (m.ida_quadr_sensi, m.ida_rhsQSDQ)
    };
    if quadr_sensi && rhsQSDQ {
        let token: Box<dyn Any> = Box::new(IDA_mem.clone());
        IDA_mem.borrow_mut().ida_user_dataQS = Some(token);
        /* Test if we have the problem parameters */
        if IDA_mem.borrow().ida_p.is_none() {
            IDAProcessError(
                Some(IDA_mem),
                IDA_ILL_INPUT,
                line!() as i32,
                "IDASolve",
                file!(),
                MSG_NULL_P,
            );
            return IDA_ILL_INPUT;
        }
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

        let ier = IDAQuadSetup(IDA_mem);
        if ier != IDA_SUCCESS {
            return ier;
        }

        /* On first call, check for tout - tn too small, set initial hh,
        check for approach to tstop, and scale phi[1], phiQ[1], and phiS[1] by hh.
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
            let mut ypnorm = IDAWrmsNorm(IDA_mem, &phi1, &ewt, suppressalg);

            if IDA_mem.borrow().ida_errconQ {
                let (phiQ1, ewtQ) = {
                    let m = IDA_mem.borrow();
                    (m.ida_phiQ[1].clone().unwrap(), m.ida_ewtQ.clone().unwrap())
                };
                ypnorm = IDAQuadWrmsNormUpdate(IDA_mem, ypnorm, &phiQ1, &ewtQ);
            }
            if IDA_mem.borrow().ida_errconS {
                let (phiS1, ewtS) = {
                    let m = IDA_mem.borrow();
                    (m.ida_phiS[1].clone(), m.ida_ewtS.clone())
                };
                ypnorm = IDASensWrmsNormUpdate(IDA_mem, ypnorm, &phiS1, &ewtS, suppressalg);
            }
            if IDA_mem.borrow().ida_errconQS {
                let (phiQS1, ewtQS) = {
                    let m = IDA_mem.borrow();
                    (m.ida_phiQS[1].clone(), m.ida_ewtQS.clone())
                };
                ypnorm = IDAQuadSensWrmsNormUpdate(IDA_mem, ypnorm, &phiQS1, &ewtQS);
            }

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

        /* set phiQ[1] = hh*yQ' */
        if IDA_mem.borrow().ida_quadr {
            let phiQ1 = IDA_mem.borrow().ida_phiQ[1].clone().unwrap();
            N_VScale(hh, &phiQ1, &phiQ1);
        }

        let (sensi, quadr_sensi) = {
            let m = IDA_mem.borrow();
            (m.ida_sensi, m.ida_quadr_sensi)
        };
        if sensi || quadr_sensi {
            let mut m = IDA_mem.borrow_mut();
            let Ns = m.ida_Ns;
            for is in 0..Ns as usize {
                m.ida_cvals[is] = hh;
            }
        }

        if sensi {
            /* set phiS[1][i] = hh*yS_i' */
            let (Ns, cvals, phiS1) = {
                let m = IDA_mem.borrow();
                (m.ida_Ns, m.ida_cvals.clone(), m.ida_phiS[1].clone())
            };
            let ier = N_VScaleVectorArray(Ns, &cvals, &phiS1, &phiS1);
            if ier != IDA_SUCCESS {
                return IDA_VECTOROP_ERR;
            }
        }

        if quadr_sensi {
            let (Ns, cvals, phiQS1) = {
                let m = IDA_mem.borrow();
                (m.ida_Ns, m.ida_cvals.clone(), m.ida_phiQS[1].clone())
            };
            let ier = N_VScaleVectorArray(Ns, &cvals, &phiQS1, &phiQS1);
            if ier != IDA_SUCCESS {
                return IDA_VECTOROP_ERR;
            }
        }

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

        /* Reset and check ewt, ewtQ, ewtS and ewtQS (if not first call). */

        if IDA_mem.borrow().ida_nst > 0 {
            /* C: ier = IDA_mem->ida_efun(phi[0], ewt, IDA_mem->ida_edata).
            `ida_edata` aliases `ida_user_data` when the user supplied
            `efun` and points at IDA_mem otherwise (IDAInitialSetup); box
            aliasing is impossible in safe Rust, so the user-efun case
            passes the CURRENT `ida_user_data` (accepted deviation class 6)
            and the default case passes the module-owned `ida_edata`
            token. The token is taken out around the call and restored on
            every path so the callback can re-enter the mem. */
            let (phi0, ewt) = {
                let m = IDA_mem.borrow();
                (m.ida_phi[0].clone().unwrap(), m.ida_ewt.clone().unwrap())
            };
            let (efun, user_efun) = {
                let m = IDA_mem.borrow();
                (m.ida_efun.expect("ida_efun set"), m.ida_user_efun)
            };
            let ier = if user_efun {
                let mut data = IDA_mem.borrow_mut().ida_user_data.take();
                let retval = efun(&phi0, &ewt, &mut data);
                IDA_mem.borrow_mut().ida_user_data = data;
                retval
            } else {
                let mut data = IDA_mem.borrow_mut().ida_edata.take();
                let retval = efun(&phi0, &ewt, &mut data);
                IDA_mem.borrow_mut().ida_edata = data;
                retval
            };

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

            let (quadr, errconQ) = {
                let m = IDA_mem.borrow();
                (m.ida_quadr, m.ida_errconQ)
            };
            if quadr && errconQ {
                let (phiQ0, ewtQ) = {
                    let m = IDA_mem.borrow();
                    (m.ida_phiQ[0].clone().unwrap(), m.ida_ewtQ.clone().unwrap())
                };
                let ier = IDAQuadEwtSet(IDA_mem, &phiQ0, &ewtQ);
                if ier != 0 {
                    let tn = IDA_mem.borrow().ida_tn;
                    IDAProcessError(
                        Some(IDA_mem),
                        IDA_ILL_INPUT,
                        line!() as i32,
                        "IDASolve",
                        file!(),
                        &MSG_EWTQ_NOW_BAD(tn),
                    );
                    istate = IDA_ILL_INPUT;
                    let _ = IDAGetSolution(IDA_mem, tn, yret, ypret);
                    IDA_mem.borrow_mut().ida_tretlast = tn;
                    *tret = tn;
                    break;
                }
            }

            if IDA_mem.borrow().ida_sensi {
                let (phiS0, ewtS) = {
                    let m = IDA_mem.borrow();
                    (m.ida_phiS[0].clone(), m.ida_ewtS.clone())
                };
                let ier = IDASensEwtSet(IDA_mem, &phiS0, &ewtS);
                if ier != 0 {
                    let tn = IDA_mem.borrow().ida_tn;
                    IDAProcessError(
                        Some(IDA_mem),
                        IDA_ILL_INPUT,
                        line!() as i32,
                        "IDASolve",
                        file!(),
                        &MSG_EWTS_NOW_BAD(tn),
                    );
                    istate = IDA_ILL_INPUT;
                    let _ = IDAGetSolution(IDA_mem, tn, yret, ypret);
                    IDA_mem.borrow_mut().ida_tretlast = tn;
                    *tret = tn;
                    break;
                }
            }

            let (quadr_sensi, errconQS) = {
                let m = IDA_mem.borrow();
                (m.ida_quadr_sensi, m.ida_errconQS)
            };
            if quadr_sensi && errconQS {
                let (phiQS0, ewtQS) = {
                    let m = IDA_mem.borrow();
                    (m.ida_phiQS[0].clone(), m.ida_ewtQS.clone())
                };
                let ier = IDAQuadSensEwtSet(IDA_mem, &phiQS0, &ewtQS);
                if ier != 0 {
                    let tn = IDA_mem.borrow().ida_tn;
                    IDAProcessError(
                        Some(IDA_mem),
                        IDA_ILL_INPUT,
                        line!() as i32,
                        "IDASolve",
                        file!(),
                        &MSG_EWTQS_NOW_BAD(tn),
                    );
                    istate = IDA_ILL_INPUT;
                    let _ = IDAGetSolution(IDA_mem, tn, yret, ypret);
                    IDA_mem.borrow_mut().ida_tretlast = tn;
                    *tret = tn;
                    break;
                }
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
            let mut nrm = IDAWrmsNorm(IDA_mem, &phi0, &ewt, suppressalg);
            if IDA_mem.borrow().ida_errconQ {
                let (phiQ0, ewtQ) = {
                    let m = IDA_mem.borrow();
                    (m.ida_phiQ[0].clone().unwrap(), m.ida_ewtQ.clone().unwrap())
                };
                nrm = IDAQuadWrmsNormUpdate(IDA_mem, nrm, &phiQ0, &ewtQ);
            }
            if IDA_mem.borrow().ida_errconS {
                let (phiS0, ewtS) = {
                    let m = IDA_mem.borrow();
                    (m.ida_phiS[0].clone(), m.ida_ewtS.clone())
                };
                nrm = IDASensWrmsNormUpdate(IDA_mem, nrm, &phiS0, &ewtS, suppressalg);
            }
            if IDA_mem.borrow().ida_errconQS {
                let (phiQS0, ewtQS) = {
                    let m = IDA_mem.borrow();
                    (m.ida_phiQS[0].clone(), m.ida_ewtQS.clone())
                };
                nrm = IDAQuadSensWrmsNormUpdate(IDA_mem, nrm, &phiQS0, &ewtQS);
            }

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
/* =====================================================================
 * idas.c FRAGMENT B — every function whose definition starts in lines
 * 3000..6000 of `src/idas/idas.c`, i.e. the 49 definitions from
 * `IDAGetDky` (idas.c:3102) through `IDAStep` (idas.c:5920..6215):
 *
 *   Interpolated output / extraction: IDAGetDky, IDAGetQuad,
 *     IDAGetQuadDky, IDAGetSens, IDAGetSensDky, IDAGetSens1,
 *     IDAGetSensDky1, IDAGetQuadSens, IDAGetQuadSensDky,
 *     IDAGetQuadSens1, IDAGetQuadSensDky1, IDAComputeY, IDAComputeYp,
 *     IDAComputeYSens, IDAComputeYpSens
 *   Deallocation: IDAFree, IDAQuadFree, IDASensFree, IDAQuadSensFree
 *   Vector alloc/free: IDACheckNvector, IDAAllocVectors, IDAFreeVectors,
 *     IDAQuadAllocVectors, IDAQuadFreeVectors, IDASensAllocVectors,
 *     IDASensFreeVectors, IDAQuadSensAllocVectors,
 *     IDAQuadSensFreeVectors
 *   Initial setup: IDAInitialSetup, IDAQuadSetup, IDAEwtSet,
 *     IDAEwtSetSS, IDAEwtSetSV, IDAQuadEwtSet, IDAQuadEwtSetSS,
 *     IDAQuadEwtSetSV, IDASensEwtSet, IDASensEwtSetEE, IDASensEwtSetSS,
 *     IDASensEwtSetSV, IDAQuadSensEwtSet, IDAQuadSensEwtSetEE,
 *     IDAQuadSensEwtSetSS, IDAQuadSensEwtSetSV
 *   Stopping tests / failure handling / step: IDAStopTest1,
 *     IDAStopTest2, IDAHandleFailure, IDAStep
 *
 * Concatenated into `idas.rs`; imports and module-scope constants come
 * from the concatenation target (`std::any::Any`, `crate::idas_impl::*`,
 * `sundials_core::sundials_{math,nvector,nonlinearsolver,types,errors}`).
 * Items outside that set are written with a fully-qualified path.
 * Everything this fragment calls but does not define lives in a sibling
 * fragment of the same module (`IDAGetSolution`, `IDASetCoeffs`,
 * `IDAPredict`, `IDASensPredict`, `IDANls`, `IDAQuadNls`, `IDASensNls`,
 * `IDAQuadSensNls`, `IDACheckConstraints`, `IDA*TestError`, `IDARestore`,
 * `IDAHandleNFlag`, `IDAReset`, `IDACompleteStep` — all in fragment C) or
 * in another crate module (`crate::idas_nls::idaNlsInit`,
 * `crate::idas_nls_sim::idaNlsInitSensSim`,
 * `crate::idas_nls_stg::idaNlsInitSensStg`, `crate::idaa::IDAAdjFree`).
 *
 * Reference build: SUNDIALS_LOGGING_LEVEL = 2 (SUNLogInfo/SUNLogInfoIf/
 * SUNLogDebug/SUNLogExtraDebug* omitted at translation time; IDA_WARNING
 * paths kept), profiling off (`SUNDIALS_MARK_FUNCTION_BEGIN/END`
 * omitted), error checks off, monitoring ON, serial branches only.
 *
 * Borrow discipline: internal functions take `&IDAMem` and use granular
 * borrows — no borrow of the mem is held across a user callback, an
 * `IDAProcessError` call, a linear/nonlinear solver call, or an
 * `N_Vector` operation on a user-visible vector.
 *
 * Error-report accounting (fidelity check): the C range contains 92
 * `IDAProcessError` call sites; exactly 24 of them are the NULL-pointer
 * guards this port drops because the handle/reference types make them
 * unrepresentable (`ida_mem == NULL` in each of the 15 public entry
 * points above, plus the `dky`/`dkyQ`/`dkyS`/`dkyQS`/`yySout`/`yyQSout`/
 * `dkySout`/`dkyQSout`/`yyQSret` NULL checks).  All 68 remaining call
 * sites are reproduced here with the same code, message and ordering.
 * =====================================================================*/

/*
 * =================================================================
 * Callback invocation helpers (the box token is taken out of the mem
 * around every user callback call and restored on EVERY path).  Named
 * `idab_*` so this fragment never collides with identically shaped
 * helpers in a sibling fragment; the integrator may dedupe them.
 * =================================================================
 */

/// Invoke the error-weight function
/// (C: `IDA_mem->ida_efun(ycur, weight, IDA_mem->ida_edata)`).
///
/// C aliases `ida_edata` with `ida_user_data` when the user supplied
/// `efun` and with `IDA_mem` otherwise (`IDAInitialSetup`).  Box aliasing
/// is impossible in safe Rust, so the user-efun case forwards the CURRENT
/// `ida_user_data` (ARCHITECTURE deviation class 6) and the default case
/// forwards the module-owned `ida_edata` token.
fn idab_call_efun(IDA_mem: &IDAMem, ycur: &N_Vector, weight: &N_Vector) -> i32 {
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

/// Invoke the user residual `res`
/// (C: `IDA_mem->ida_res(tt, yy, yp, rr, IDA_mem->ida_user_data)`).
fn idab_call_res(
    IDA_mem: &IDAMem,
    tt: sunrealtype,
    yy: &N_Vector,
    yp: &N_Vector,
    rr: &N_Vector,
) -> i32 {
    let res = IDA_mem.borrow().ida_res.expect("ida_res set");
    let mut user_data = IDA_mem.borrow_mut().ida_user_data.take();
    let retval = res(tt, yy, yp, rr, &mut user_data);
    IDA_mem.borrow_mut().ida_user_data = user_data;
    retval
}

/// Invoke the quadrature RHS `rhsQ` (C:
/// `IDA_mem->ida_rhsQ(tres, yy, yp, rrQ, IDA_mem->ida_user_data)` — every
/// `ida_rhsQ` call site in `idas.c` passes `ida_user_data`, never
/// `ida_user_dataQ`).
fn idab_call_rhsQ(
    IDA_mem: &IDAMem,
    tres: sunrealtype,
    yy: &N_Vector,
    yp: &N_Vector,
    rrQ: &N_Vector,
) -> i32 {
    let rhsQ = IDA_mem.borrow().ida_rhsQ.expect("ida_rhsQ set");
    let mut user_data = IDA_mem.borrow_mut().ida_user_data.take();
    let retval = rhsQ(tres, yy, yp, rrQ, &mut user_data);
    IDA_mem.borrow_mut().ida_user_data = user_data;
    retval
}

/// Invoke the quadrature-sensitivity RHS `rhsQS` (C:
/// `IDA_mem->ida_rhsQS(Ns, t, yy, yp, yyS, ypS, rrQ, rhsvalQS,
/// IDA_mem->ida_user_dataQS, yytmp, yptmp, tmpQS)`).
///
/// `ida_user_dataQS` is `Some(token)` when IDAS uses its internal DQ
/// routine (C stored `IDA_mem` there) and `None` when C stored
/// `ida_user_data`; the `None` case therefore forwards the integrator's
/// `ida_user_data` box.
#[allow(clippy::too_many_arguments)]
fn idab_call_rhsQS(
    IDA_mem: &IDAMem,
    Ns: i32,
    t: sunrealtype,
    yy: &N_Vector,
    yp: &N_Vector,
    yyS: &[N_Vector],
    ypS: &[N_Vector],
    rrQ: &N_Vector,
    rhsvalQS: &[N_Vector],
    yytmp: &N_Vector,
    yptmp: &N_Vector,
    tmpQS: &N_Vector,
) -> i32 {
    let rhsQS = IDA_mem.borrow().ida_rhsQS.expect("ida_rhsQS set");
    let mut token = IDA_mem.borrow_mut().ida_user_dataQS.take();
    if token.is_some() {
        let retval = rhsQS(
            Ns, t, yy, yp, yyS, ypS, rrQ, rhsvalQS, &mut token, yytmp, yptmp, tmpQS,
        );
        IDA_mem.borrow_mut().ida_user_dataQS = token;
        retval
    } else {
        let mut user_data = IDA_mem.borrow_mut().ida_user_data.take();
        let retval = rhsQS(
            Ns,
            t,
            yy,
            yp,
            yyS,
            ypS,
            rrQ,
            rhsvalQS,
            &mut user_data,
            yytmp,
            yptmp,
            tmpQS,
        );
        let mut m = IDA_mem.borrow_mut();
        m.ida_user_data = user_data;
        m.ida_user_dataQS = token;
        retval
    }
}

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
 * IDAGetQuad
 *
 * The following function can be called to obtain the quadrature
 * variables after a successful integration step.
 *
 * This is just a wrapper that calls IDAGetQuadDky with k=0.
 */

pub fn IDAGetQuad(ida_mem: &IDAMem, ptret: &mut sunrealtype, yQout: &N_Vector) -> i32 {
    /* NULL-mem check: handled by type system */
    let IDA_mem = ida_mem;

    let tretlast = IDA_mem.borrow().ida_tretlast;
    *ptret = tretlast;

    IDAGetQuadDky(ida_mem, tretlast, 0, yQout)
}

/*
 * IDAGetQuadDky
 *
 * Returns the quadrature variables (or their
 * derivatives up to the current method order) at any time within
 * the last integration step (dense output).
 */
pub fn IDAGetQuadDky(ida_mem: &IDAMem, t: sunrealtype, k: i32, dkyQ: &N_Vector) -> i32 {
    /* Check ida_mem */
    /* NULL-mem check: handled by type system */
    let IDA_mem = ida_mem;

    /* Check if quadrature was initialized */
    if IDA_mem.borrow().ida_quadr != SUNTRUE {
        IDAProcessError(
            Some(IDA_mem),
            IDA_NO_QUAD,
            line!() as i32,
            "IDAGetQuadDky",
            file!(),
            MSG_NO_QUAD,
        );
        return IDA_NO_QUAD;
    }

    /* NULL dkyQ check: handled by type system */

    if (k < 0) || (k > IDA_mem.borrow().ida_kk) {
        IDAProcessError(
            Some(IDA_mem),
            IDA_BAD_K,
            line!() as i32,
            "IDAGetQuadDky",
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

    /* NOTE: upstream computes tfuzz here WITHOUT SUNRabs and without the
    negative-h sign flip used elsewhere — preserved verbatim. */
    let tfuzz = HUNDRED * uround * (tn + hh);
    let tp = tn - hused - tfuzz;
    if (t - tp) * hh < ZERO {
        IDAProcessError(
            Some(IDA_mem),
            IDA_BAD_T,
            line!() as i32,
            "IDAGetQuadDky",
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
        if i == 0 {
            cjk[i as usize] = ONE;
            psij_1 = ZERO;
        } else {
            cjk[i as usize] = cjk[(i - 1) as usize] * (i as sunrealtype) / psi[(i - 1) as usize];
            psij_1 = psi[(i - 1) as usize];
        }

        /* update c_j^(i) */
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

    let Xvecs: Vec<N_Vector> = {
        let m = IDA_mem.borrow();
        ((k as usize)..=(kused as usize))
            .map(|j| m.ida_phiQ[j].clone().unwrap())
            .collect()
    };
    let retval = N_VLinearCombination(kused - k + 1, &cjk[(k as usize)..], &Xvecs, dkyQ);
    if retval != IDA_SUCCESS {
        return IDA_VECTOROP_ERR;
    }

    IDA_SUCCESS
}

/*
 * IDAGetSens
 *
 * This routine extracts sensitivity solution into yySout at the
 * time at which IDASolve returned the solution.
 * This is just a wrapper that calls IDAGetSensDky1 with k=0 and
 * is=0, 1, ... ,NS-1.
 */

pub fn IDAGetSens(ida_mem: &IDAMem, ptret: &mut sunrealtype, yySout: &[N_Vector]) -> i32 {
    let mut ierr: i32 = 0;

    /* Check ida_mem */
    /* NULL-mem check: handled by type system */
    let IDA_mem = ida_mem;

    /*Check the parameters */
    /* NULL yySout check: handled by type system */

    /* are sensitivities enabled? */
    if IDA_mem.borrow().ida_sensi == SUNFALSE {
        IDAProcessError(
            Some(IDA_mem),
            IDA_NO_SENS,
            line!() as i32,
            "IDAGetSens",
            file!(),
            MSG_NO_SENSI,
        );
        return IDA_NO_SENS;
    }

    *ptret = IDA_mem.borrow().ida_tretlast;

    let Ns = IDA_mem.borrow().ida_Ns;
    for is in 0..Ns {
        ierr = IDAGetSensDky1(ida_mem, *ptret, 0, is, &yySout[is as usize]);
        if IDA_SUCCESS != ierr {
            break;
        }
    }

    ierr
}

/*
 * IDAGetSensDky
 *
 * Computes the k-th derivative of all sensitivities of the y function at
 * time t. It repeatedly calls IDAGetSensDky1. The argument dkyS must be
 * a pointer to N_Vector and must be allocated by the user to hold at
 * least Ns vectors.
 */
pub fn IDAGetSensDky(ida_mem: &IDAMem, t: sunrealtype, k: i32, dkySout: &[N_Vector]) -> i32 {
    let mut ier: i32 = 0;

    /* Check all inputs for legality */

    /* NULL-mem check: handled by type system */
    let IDA_mem = ida_mem;

    if IDA_mem.borrow().ida_sensi == SUNFALSE {
        IDAProcessError(
            Some(IDA_mem),
            IDA_NO_SENS,
            line!() as i32,
            "IDAGetSensDky",
            file!(),
            MSG_NO_SENSI,
        );
        return IDA_NO_SENS;
    }

    /* NULL dkySout check: handled by type system */

    if (k < 0) || (k > IDA_mem.borrow().ida_kk) {
        IDAProcessError(
            Some(IDA_mem),
            IDA_BAD_K,
            line!() as i32,
            "IDAGetSensDky",
            file!(),
            MSG_BAD_K,
        );
        return IDA_BAD_K;
    }

    let Ns = IDA_mem.borrow().ida_Ns;
    for is in 0..Ns {
        ier = IDAGetSensDky1(ida_mem, t, k, is, &dkySout[is as usize]);
        if ier != IDA_SUCCESS {
            break;
        }
    }

    ier
}

/*
 * IDAGetSens1
 *
 * This routine extracts the is-th sensitivity solution into ySout
 * at the time at which IDASolve returned the solution.
 * This is just a wrapper that calls IDASensDky1 with k=0.
 */

pub fn IDAGetSens1(ida_mem: &IDAMem, ptret: &mut sunrealtype, is: i32, yySret: &N_Vector) -> i32 {
    /* NULL-mem check: handled by type system */
    let IDA_mem = ida_mem;

    let tretlast = IDA_mem.borrow().ida_tretlast;
    *ptret = tretlast;

    IDAGetSensDky1(ida_mem, tretlast, 0, is, yySret)
}

/*
 * IDAGetSensDky1
 *
 * IDASensDky1 computes the kth derivative of the yS[is] function
 * at time t, where tn-hu <= t <= tn, tn denotes the current
 * internal time reached, and hu is the last internal step size
 * successfully used by the solver. The user may request
 * is=0, 1, ..., Ns-1 and k=0, 1, ..., kk, where kk is the current
 * order. The derivative vector is returned in dky. This vector
 * must be allocated by the caller. It is only legal to call this
 * function after a successful return from IDASolve with sensitivity
 * computation enabled.
 */
pub fn IDAGetSensDky1(ida_mem: &IDAMem, t: sunrealtype, k: i32, is: i32, dkyS: &N_Vector) -> i32 {
    /* Check all inputs for legality */
    /* NULL-mem check: handled by type system */
    let IDA_mem = ida_mem;

    if IDA_mem.borrow().ida_sensi == SUNFALSE {
        IDAProcessError(
            Some(IDA_mem),
            IDA_NO_SENS,
            line!() as i32,
            "IDAGetSensDky1",
            file!(),
            MSG_NO_SENSI,
        );
        return IDA_NO_SENS;
    }

    /* NULL dkyS check: handled by type system */

    /* Is the requested sensitivity index valid? */
    if is < 0 || is >= IDA_mem.borrow().ida_Ns {
        IDAProcessError(
            Some(IDA_mem),
            IDA_BAD_IS,
            line!() as i32,
            "IDAGetSensDky1",
            file!(),
            MSG_BAD_IS,
        );
        return IDA_BAD_IS;
    }

    /* Is the requested order valid? */
    if (k < 0) || (k > IDA_mem.borrow().ida_kused) {
        IDAProcessError(
            Some(IDA_mem),
            IDA_BAD_K,
            line!() as i32,
            "IDAGetSensDky1",
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
            "IDAGetSensDky1",
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
        if i == 0 {
            cjk[i as usize] = ONE;
            psij_1 = ZERO;
        } else {
            cjk[i as usize] = cjk[(i - 1) as usize] * (i as sunrealtype) / psi[(i - 1) as usize];
            psij_1 = psi[(i - 1) as usize];
        }

        /* Update cjk based on the recurrence */
        for j in (i + 1)..=(kused - k + i) {
            cjk[j as usize] = ((i as sunrealtype) * cjk_1[(j - 1) as usize]
                + cjk[(j - 1) as usize] * (delt + psij_1))
                / psi[(j - 1) as usize];
            psij_1 = psi[(j - 1) as usize];
        }

        /* Update cjk_1 for the next step */
        for j in (i + 1)..=(kused - k + i) {
            cjk_1[j as usize] = cjk[j as usize];
        }
    }

    /* Compute sum (c_j(t) * phi(t)) */
    /* `ida_Xvecs` is C scratch for the handle array; the locked port
    pattern rebuilds it on demand (an N_Vector array cannot be left
    uninitialized in safe Rust) — see `cv_Xvecs` in cvodes.rs. */
    let Xvecs: Vec<N_Vector> = {
        let m = IDA_mem.borrow();
        ((k as usize)..=(kused as usize))
            .map(|j| m.ida_phiS[j][is as usize].clone())
            .collect()
    };

    let retval = N_VLinearCombination(kused - k + 1, &cjk[(k as usize)..], &Xvecs, dkyS);
    if retval != IDA_SUCCESS {
        return IDA_VECTOROP_ERR;
    }

    IDA_SUCCESS
}

/*
 * IDAGetQuadSens
 *
 * This routine extracts quadrature sensitivity solution into yyQSout at the
 * time at which IDASolve returned the solution.
 * This is just a wrapper that calls IDAGetQuadSensDky1 with k=0 and
 * is=0, 1, ... ,NS-1.
 */

pub fn IDAGetQuadSens(ida_mem: &IDAMem, ptret: &mut sunrealtype, yyQSout: &[N_Vector]) -> i32 {
    let mut ierr: i32 = 0;

    /* Check ida_mem */
    /* NULL-mem check: handled by type system */
    let IDA_mem = ida_mem;

    /*Check the parameters */
    /* NULL yyQSout check: handled by type system */

    /* are sensitivities enabled? */
    if IDA_mem.borrow().ida_quadr_sensi == SUNFALSE {
        IDAProcessError(
            Some(IDA_mem),
            IDA_NO_SENS,
            line!() as i32,
            "IDAGetQuadSens",
            file!(),
            MSG_NO_QUADSENSI,
        );
        return IDA_NO_SENS;
    }

    *ptret = IDA_mem.borrow().ida_tretlast;

    let Ns = IDA_mem.borrow().ida_Ns;
    for is in 0..Ns {
        ierr = IDAGetQuadSensDky1(ida_mem, *ptret, 0, is, &yyQSout[is as usize]);
        if IDA_SUCCESS != ierr {
            break;
        }
    }

    ierr
}

/*
 * IDAGetQuadSensDky
 *
 * Computes the k-th derivative of all quadratures sensitivities of the y function at
 * time t. It repeatedly calls IDAGetQuadSensDky. The argument dkyS must be
 * a pointer to N_Vector and must be allocated by the user to hold at
 * least Ns vectors.
 */
pub fn IDAGetQuadSensDky(ida_mem: &IDAMem, t: sunrealtype, k: i32, dkyQSout: &[N_Vector]) -> i32 {
    let mut ier: i32 = 0;

    /* Check all inputs for legality */

    /* NULL-mem check: handled by type system */
    let IDA_mem = ida_mem;

    if IDA_mem.borrow().ida_sensi == SUNFALSE {
        IDAProcessError(
            Some(IDA_mem),
            IDA_NO_SENS,
            line!() as i32,
            "IDAGetQuadSensDky",
            file!(),
            MSG_NO_SENSI,
        );
        return IDA_NO_SENS;
    }

    if IDA_mem.borrow().ida_quadr_sensi == SUNFALSE {
        IDAProcessError(
            Some(IDA_mem),
            IDA_NO_QUADSENS,
            line!() as i32,
            "IDAGetQuadSensDky",
            file!(),
            MSG_NO_QUADSENSI,
        );
        return IDA_NO_QUADSENS;
    }

    /* NULL dkyQSout check: handled by type system */

    if (k < 0) || (k > IDA_mem.borrow().ida_kk) {
        IDAProcessError(
            Some(IDA_mem),
            IDA_BAD_K,
            line!() as i32,
            "IDAGetQuadSensDky",
            file!(),
            MSG_BAD_K,
        );
        return IDA_BAD_K;
    }

    let Ns = IDA_mem.borrow().ida_Ns;
    for is in 0..Ns {
        ier = IDAGetQuadSensDky1(ida_mem, t, k, is, &dkyQSout[is as usize]);
        if ier != IDA_SUCCESS {
            break;
        }
    }

    ier
}

/*
 * IDAGetQuadSens1
 *
 * This routine extracts the is-th quadrature sensitivity solution into yQSout
 * at the time at which IDASolve returned the solution.
 * This is just a wrapper that calls IDASensDky1 with k=0.
 */

pub fn IDAGetQuadSens1(
    ida_mem: &IDAMem,
    ptret: &mut sunrealtype,
    is: i32,
    yyQSret: &N_Vector,
) -> i32 {
    /* NULL-mem check: handled by type system */
    let IDA_mem = ida_mem;

    if IDA_mem.borrow().ida_sensi == SUNFALSE {
        IDAProcessError(
            Some(IDA_mem),
            IDA_NO_SENS,
            line!() as i32,
            "IDAGetQuadSens1",
            file!(),
            MSG_NO_SENSI,
        );
        return IDA_NO_SENS;
    }

    if IDA_mem.borrow().ida_quadr_sensi == SUNFALSE {
        IDAProcessError(
            Some(IDA_mem),
            IDA_NO_QUADSENS,
            line!() as i32,
            "IDAGetQuadSens1",
            file!(),
            MSG_NO_QUADSENSI,
        );
        return IDA_NO_QUADSENS;
    }

    /* NULL yyQSret check: handled by type system */

    let tretlast = IDA_mem.borrow().ida_tretlast;
    *ptret = tretlast;

    IDAGetQuadSensDky1(ida_mem, tretlast, 0, is, yyQSret)
}

/*
 * IDAGetQuadSensDky1
 *
 * IDAGetQuadSensDky1 computes the kth derivative of the yS[is] function
 * at time t, where tn-hu <= t <= tn, tn denotes the current
 * internal time reached, and hu is the last internal step size
 * successfully used by the solver. The user may request
 * is=0, 1, ..., Ns-1 and k=0, 1, ..., kk, where kk is the current
 * order. The derivative vector is returned in dky. This vector
 * must be allocated by the caller. It is only legal to call this
 * function after a successful return from IDASolve with sensitivity
 * computation enabled.
 */
pub fn IDAGetQuadSensDky1(
    ida_mem: &IDAMem,
    t: sunrealtype,
    k: i32,
    is: i32,
    dkyQS: &N_Vector,
) -> i32 {
    /* Check all inputs for legality */
    /* NULL-mem check: handled by type system */
    let IDA_mem = ida_mem;

    if IDA_mem.borrow().ida_sensi == SUNFALSE {
        IDAProcessError(
            Some(IDA_mem),
            IDA_NO_SENS,
            line!() as i32,
            "IDAGetQuadSensDky1",
            file!(),
            MSG_NO_SENSI,
        );
        return IDA_NO_SENS;
    }

    if IDA_mem.borrow().ida_quadr_sensi == SUNFALSE {
        IDAProcessError(
            Some(IDA_mem),
            IDA_NO_QUADSENS,
            line!() as i32,
            "IDAGetQuadSensDky1",
            file!(),
            MSG_NO_QUADSENSI,
        );
        return IDA_NO_QUADSENS;
    }

    /* NULL dkyQS check: handled by type system */

    /* Is the requested sensitivity index valid*/
    if is < 0 || is >= IDA_mem.borrow().ida_Ns {
        IDAProcessError(
            Some(IDA_mem),
            IDA_BAD_IS,
            line!() as i32,
            "IDAGetQuadSensDky1",
            file!(),
            MSG_BAD_IS,
        );
        return IDA_BAD_IS;
    }

    /* Is the requested order valid? */
    if (k < 0) || (k > IDA_mem.borrow().ida_kused) {
        IDAProcessError(
            Some(IDA_mem),
            IDA_BAD_K,
            line!() as i32,
            "IDAGetQuadSensDky1",
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
            "IDAGetQuadSensDky1",
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
        if i == 0 {
            cjk[i as usize] = ONE;
            psij_1 = ZERO;
        } else {
            cjk[i as usize] = cjk[(i - 1) as usize] * (i as sunrealtype) / psi[(i - 1) as usize];
            psij_1 = psi[(i - 1) as usize];
        }

        /* Update cjk based on the recurrence */
        for j in (i + 1)..=(kused - k + i) {
            cjk[j as usize] = ((i as sunrealtype) * cjk_1[(j - 1) as usize]
                + cjk[(j - 1) as usize] * (delt + psij_1))
                / psi[(j - 1) as usize];
            psij_1 = psi[(j - 1) as usize];
        }

        /* Update cjk_1 for the next step */
        for j in (i + 1)..=(kused - k + i) {
            cjk_1[j as usize] = cjk[j as usize];
        }
    }

    /* Compute sum (c_j(t) * phi(t)) */
    let Xvecs: Vec<N_Vector> = {
        let m = IDA_mem.borrow();
        ((k as usize)..=(kused as usize))
            .map(|j| m.ida_phiQS[j][is as usize].clone())
            .collect()
    };

    let retval = N_VLinearCombination(kused - k + 1, &cjk[(k as usize)..], &Xvecs, dkyQS);
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
 * IDAComputeYSens
 *
 * Computes yS based on the current prediction and given correction.
 */
pub fn IDAComputeYSens(ida_mem: &IDAMem, ycorS: &[N_Vector], yyS: &[N_Vector]) -> i32 {
    /* NULL-mem check: handled by type system */
    let IDA_mem = ida_mem;

    let (Ns, yySpredict) = {
        let m = IDA_mem.borrow();
        (m.ida_Ns, m.ida_yySpredict.clone())
    };
    let _ = N_VLinearSumVectorArray(Ns, ONE, &yySpredict, ONE, ycorS, yyS);

    IDA_SUCCESS
}

/*
 * IDAComputeYpSens
 *
 * Computes yS' based on the current prediction and given correction.
 */
pub fn IDAComputeYpSens(ida_mem: &IDAMem, ycorS: &[N_Vector], ypS: &[N_Vector]) -> i32 {
    /* NULL-mem check: handled by type system */
    let IDA_mem = ida_mem;

    let (Ns, ypSpredict, cj) = {
        let m = IDA_mem.borrow();
        (m.ida_Ns, m.ida_ypSpredict.clone(), m.ida_cj)
    };
    let _ = N_VLinearSumVectorArray(Ns, ONE, &ypSpredict, cj, ycorS, ypS);

    IDA_SUCCESS
}

/*
 * -----------------------------------------------------------------
 * Deallocation functions
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

    IDAQuadFree(&IDA_mem);

    IDASensFree(&IDA_mem);

    IDAQuadSensFree(&IDA_mem);

    crate::idaa::IDAAdjFree(&IDA_mem);

    /* if IDA created the NLS object then free it */
    if IDA_mem.borrow().ownNLS {
        let nls = {
            let mut m = IDA_mem.borrow_mut();
            m.ownNLS = SUNFALSE;
            m.NLS.take()
        };
        let _ = sundials_core::sundials_nonlinearsolver::SUNNonlinSolFree(nls);
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

    {
        let mut m = IDA_mem.borrow_mut();
        m.ida_cvals = Vec::new();
        m.ida_Xvecs = Vec::new();
        m.ida_Zvecs = Vec::new();
    }

    /* C `IDA_mem->python = NULL` omitted — the Python bindings field is
    out of scope and absent from IDAMemRec. */

    /* C frees the mem struct wholesale; the Rust handle is dropped by the
    caller, so break the Rc cycles the module-owned callback tokens create
    (ida_edata / ida_user_dataS / ida_user_dataQS hold IDAMem clones
    pointing back at this record) */
    {
        let mut m = IDA_mem.borrow_mut();
        m.ida_edata = None;
        m.ida_user_dataS = None;
        m.ida_user_dataQS = None;
    }

    *ida_mem = None;
}

/*
 * IDAQuadFree
 *
 * IDAQuadFree frees the problem memory in ida_mem allocated
 * for quadrature integration. Its only argument is the pointer
 * ida_mem returned by IDACreate.
 */

pub fn IDAQuadFree(ida_mem: &IDAMem) {
    /* NULL-mem check: handled by type system */
    let IDA_mem = ida_mem;

    if IDA_mem.borrow().ida_quadMallocDone {
        IDAQuadFreeVectors(IDA_mem);
        let mut m = IDA_mem.borrow_mut();
        m.ida_quadMallocDone = SUNFALSE;
        m.ida_quadr = SUNFALSE;
    }
}

/*
 * IDASensFree
 *
 * IDASensFree frees the problem memory in ida_mem allocated
 * for sensitivity analysis. Its only argument is the pointer
 * ida_mem returned by IDACreate.
 */

pub fn IDASensFree(ida_mem: &IDAMem) {
    /* return immediately if IDA memory is NULL */
    /* NULL-mem check: handled by type system */
    let IDA_mem = ida_mem;

    if IDA_mem.borrow().ida_sensMallocDone {
        IDASensFreeVectors(IDA_mem);
        let mut m = IDA_mem.borrow_mut();
        m.ida_sensMallocDone = SUNFALSE;
        m.ida_sensi = SUNFALSE;
    }

    /* free any vector wrappers */
    if IDA_mem.borrow().simMallocDone {
        let (ypredictSim, ycorSim, ewtSim) = {
            let mut m = IDA_mem.borrow_mut();
            let v = (m.ypredictSim.take(), m.ycorSim.take(), m.ewtSim.take());
            m.simMallocDone = SUNFALSE;
            v
        };
        if let Some(v) = ypredictSim {
            N_VDestroy(v);
        }
        if let Some(v) = ycorSim {
            N_VDestroy(v);
        }
        if let Some(v) = ewtSim {
            N_VDestroy(v);
        }
    }
    if IDA_mem.borrow().stgMallocDone {
        let (ypredictStg, ycorStg, ewtStg) = {
            let mut m = IDA_mem.borrow_mut();
            let v = (m.ypredictStg.take(), m.ycorStg.take(), m.ewtStg.take());
            m.stgMallocDone = SUNFALSE;
            v
        };
        if let Some(v) = ypredictStg {
            N_VDestroy(v);
        }
        if let Some(v) = ycorStg {
            N_VDestroy(v);
        }
        if let Some(v) = ewtStg {
            N_VDestroy(v);
        }
    }

    /* if IDA created the NLS object then free it */
    if IDA_mem.borrow().ownNLSsim {
        let nls = {
            let mut m = IDA_mem.borrow_mut();
            m.ownNLSsim = SUNFALSE;
            m.NLSsim.take()
        };
        let _ = sundials_core::sundials_nonlinearsolver::SUNNonlinSolFree(nls);
    }
    if IDA_mem.borrow().ownNLSstg {
        let nls = {
            let mut m = IDA_mem.borrow_mut();
            m.ownNLSstg = SUNFALSE;
            m.NLSstg.take()
        };
        let _ = sundials_core::sundials_nonlinearsolver::SUNNonlinSolFree(nls);
    }

    /* free min atol array if necessary */
    if !IDA_mem.borrow().ida_atolSmin0.is_empty() {
        IDA_mem.borrow_mut().ida_atolSmin0 = Vec::new();
    }
}

/*
 * IDAQuadSensFree
 *
 * IDAQuadSensFree frees the problem memory in ida_mem allocated
 * for quadrature sensitivity analysis. Its only argument is the
 * pointer ida_mem returned by IDACreate.
 */
pub fn IDAQuadSensFree(ida_mem: &IDAMem) {
    /* NULL-mem check: handled by type system */
    let IDA_mem = ida_mem;

    if IDA_mem.borrow().ida_quadSensMallocDone {
        IDAQuadSensFreeVectors(IDA_mem);
        let mut m = IDA_mem.borrow_mut();
        m.ida_quadSensMallocDone = SUNFALSE;
        m.ida_quadr_sensi = SUNFALSE;
    }

    /* free min atol array if necessary */
    if !IDA_mem.borrow().ida_atolQSmin0.is_empty() {
        IDA_mem.borrow_mut().ida_atolQSmin0 = Vec::new();
    }
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
 * IDAQuadAllocVectors
 *
 * NOTE: Space for ewtQ is allocated even when errconQ=SUNFALSE,
 * although in this case, ewtQ is never used. The reason for this
 * decision is to allow the user to re-initialize the quadrature
 * computation with errconQ=SUNTRUE, after an initialization with
 * errconQ=SUNFALSE, without new memory allocation within
 * IDAQuadReInit.
 */

fn IDAQuadAllocVectors(IDA_mem: &IDAMem, tmpl: &N_Vector) -> sunbooleantype {
    /* Allocate yyQ */
    let yyQ = match N_VClone(tmpl) {
        Some(v) => v,
        None => return SUNFALSE,
    };

    /* Allocate ypQ */
    let ypQ = match N_VClone(tmpl) {
        Some(v) => v,
        None => {
            N_VDestroy(yyQ);
            return SUNFALSE;
        }
    };

    /* Allocate ewtQ */
    let ewtQ = match N_VClone(tmpl) {
        Some(v) => v,
        None => {
            N_VDestroy(yyQ);
            N_VDestroy(ypQ);
            return SUNFALSE;
        }
    };

    /* Allocate eeQ */
    let eeQ = match N_VClone(tmpl) {
        Some(v) => v,
        None => {
            N_VDestroy(yyQ);
            N_VDestroy(ypQ);
            N_VDestroy(ewtQ);
            return SUNFALSE;
        }
    };

    {
        let mut m = IDA_mem.borrow_mut();
        m.ida_yyQ = Some(yyQ);
        m.ida_ypQ = Some(ypQ);
        m.ida_ewtQ = Some(ewtQ);
        m.ida_eeQ = Some(eeQ);
    }

    let maxord = IDA_mem.borrow().ida_maxord;
    for j in 0..=maxord as usize {
        match N_VClone(tmpl) {
            Some(v) => {
                IDA_mem.borrow_mut().ida_phiQ[j] = Some(v);
            }
            None => {
                let mut m = IDA_mem.borrow_mut();
                if let Some(v) = m.ida_yyQ.take() {
                    N_VDestroy(v);
                }
                if let Some(v) = m.ida_ypQ.take() {
                    N_VDestroy(v);
                }
                if let Some(v) = m.ida_ewtQ.take() {
                    N_VDestroy(v);
                }
                if let Some(v) = m.ida_eeQ.take() {
                    N_VDestroy(v);
                }
                for i in 0..j {
                    if let Some(v) = m.ida_phiQ[i].take() {
                        N_VDestroy(v);
                    }
                }
                return SUNFALSE;
            }
        }
    }

    {
        let mut m = IDA_mem.borrow_mut();
        m.ida_lrw += ((maxord + 4) as i64) * m.ida_lrw1Q;
        m.ida_liw += ((maxord + 4) as i64) * m.ida_liw1Q;
    }

    SUNTRUE
}

/*
 * IDAQuadFreeVectors
 *
 * This routine frees the IDAS vectors allocated in IDAQuadAllocVectors.
 */

fn IDAQuadFreeVectors(IDA_mem: &IDAMem) {
    let mut m = IDA_mem.borrow_mut();

    if let Some(v) = m.ida_yyQ.take() {
        N_VDestroy(v);
    }
    if let Some(v) = m.ida_ypQ.take() {
        N_VDestroy(v);
    }
    if let Some(v) = m.ida_ewtQ.take() {
        N_VDestroy(v);
    }
    if let Some(v) = m.ida_eeQ.take() {
        N_VDestroy(v);
    }
    let maxord = m.ida_maxord;
    for j in 0..=maxord as usize {
        if let Some(v) = m.ida_phiQ[j].take() {
            N_VDestroy(v);
        }
    }

    /* NOTE: upstream subtracts (maxord + 5) here but adds (maxord + 4) in
    IDAQuadAllocVectors — asymmetry preserved verbatim. */
    m.ida_lrw -= ((maxord + 5) as i64) * m.ida_lrw1Q;
    m.ida_liw -= ((maxord + 5) as i64) * m.ida_liw1Q;

    if m.ida_VatolQMallocDone {
        if let Some(v) = m.ida_VatolQ.take() {
            N_VDestroy(v);
        }
        m.ida_lrw -= m.ida_lrw1Q;
        m.ida_liw -= m.ida_liw1Q;
    }

    m.ida_VatolQMallocDone = SUNFALSE;
}

/*
 * IDASensAllocVectors
 *
 * Allocates space for the N_Vectors, plist, and pbar required for FSA.
 */

fn IDASensAllocVectors(IDA_mem: &IDAMem, tmpl: &N_Vector) -> sunbooleantype {
    {
        let (tempv1, tempv2) = {
            let m = IDA_mem.borrow();
            (m.ida_tempv1.clone(), m.ida_tempv2.clone())
        };
        let mut m = IDA_mem.borrow_mut();
        m.ida_tmpS1 = tempv1;
        m.ida_tmpS2 = tempv2;
    }

    let Ns = IDA_mem.borrow().ida_Ns;

    /* Allocate space for workspace vectors */

    let tmpS3 = match N_VClone(tmpl) {
        Some(v) => v,
        None => return SUNFALSE,
    };

    let ewtS = match N_VCloneVectorArray(Ns, tmpl) {
        Some(v) => v,
        None => {
            N_VDestroy(tmpS3);
            return SUNFALSE;
        }
    };

    let eeS = match N_VCloneVectorArray(Ns, tmpl) {
        Some(v) => v,
        None => {
            N_VDestroy(tmpS3);
            N_VDestroyVectorArray(ewtS, Ns);
            return SUNFALSE;
        }
    };

    let yyS = match N_VCloneVectorArray(Ns, tmpl) {
        Some(v) => v,
        None => {
            N_VDestroyVectorArray(eeS, Ns);
            N_VDestroyVectorArray(ewtS, Ns);
            N_VDestroy(tmpS3);
            return SUNFALSE;
        }
    };

    let ypS = match N_VCloneVectorArray(Ns, tmpl) {
        Some(v) => v,
        None => {
            N_VDestroyVectorArray(yyS, Ns);
            N_VDestroyVectorArray(eeS, Ns);
            N_VDestroyVectorArray(ewtS, Ns);
            N_VDestroy(tmpS3);
            return SUNFALSE;
        }
    };

    let yySpredict = match N_VCloneVectorArray(Ns, tmpl) {
        Some(v) => v,
        None => {
            N_VDestroyVectorArray(ypS, Ns);
            N_VDestroyVectorArray(yyS, Ns);
            N_VDestroyVectorArray(eeS, Ns);
            N_VDestroyVectorArray(ewtS, Ns);
            N_VDestroy(tmpS3);
            return SUNFALSE;
        }
    };

    let ypSpredict = match N_VCloneVectorArray(Ns, tmpl) {
        Some(v) => v,
        None => {
            N_VDestroyVectorArray(yySpredict, Ns);
            N_VDestroyVectorArray(ypS, Ns);
            N_VDestroyVectorArray(yyS, Ns);
            N_VDestroyVectorArray(eeS, Ns);
            N_VDestroyVectorArray(ewtS, Ns);
            N_VDestroy(tmpS3);
            return SUNFALSE;
        }
    };

    let deltaS = match N_VCloneVectorArray(Ns, tmpl) {
        Some(v) => v,
        None => {
            N_VDestroyVectorArray(ypSpredict, Ns);
            N_VDestroyVectorArray(yySpredict, Ns);
            N_VDestroyVectorArray(ypS, Ns);
            N_VDestroyVectorArray(yyS, Ns);
            N_VDestroyVectorArray(eeS, Ns);
            N_VDestroyVectorArray(ewtS, Ns);
            N_VDestroy(tmpS3);
            return SUNFALSE;
        }
    };

    {
        let mut m = IDA_mem.borrow_mut();
        m.ida_tmpS3 = Some(tmpS3);
        m.ida_ewtS = ewtS;
        m.ida_eeS = eeS;
        m.ida_yyS = yyS;
        m.ida_ypS = ypS;
        m.ida_yySpredict = yySpredict;
        m.ida_ypSpredict = ypSpredict;
        m.ida_deltaS = deltaS;

        /* Update solver workspace lengths */
        m.ida_lrw += ((5 * Ns + 1) as i64) * m.ida_lrw1;
        m.ida_liw += ((5 * Ns + 1) as i64) * m.ida_liw1;
    }

    /* Allocate space for phiS */
    /*  Make sure phiS[2], phiS[3] and phiS[4] are
    allocated (for use as temporary vectors), regardless of maxord.*/

    let maxcol = SUNMAX(IDA_mem.borrow().ida_maxord, 4);
    for j in 0..=maxcol as usize {
        match N_VCloneVectorArray(Ns, tmpl) {
            Some(v) => {
                IDA_mem.borrow_mut().ida_phiS[j] = v;
            }
            None => {
                let mut m = IDA_mem.borrow_mut();
                if let Some(v) = m.ida_tmpS3.take() {
                    N_VDestroy(v);
                }
                N_VDestroyVectorArray(std::mem::take(&mut m.ida_ewtS), Ns);
                N_VDestroyVectorArray(std::mem::take(&mut m.ida_eeS), Ns);
                N_VDestroyVectorArray(std::mem::take(&mut m.ida_yyS), Ns);
                N_VDestroyVectorArray(std::mem::take(&mut m.ida_ypS), Ns);
                N_VDestroyVectorArray(std::mem::take(&mut m.ida_yySpredict), Ns);
                N_VDestroyVectorArray(std::mem::take(&mut m.ida_ypSpredict), Ns);
                N_VDestroyVectorArray(std::mem::take(&mut m.ida_deltaS), Ns);
                /* (upstream leaks the phiS arrays allocated so far; here the
                stored handles are simply dropped with the mem record) */
                return SUNFALSE;
            }
        }
    }

    {
        let mut m = IDA_mem.borrow_mut();

        /* Update solver workspace lengths */
        m.ida_lrw += (maxcol as i64) * (Ns as i64) * m.ida_lrw1;
        m.ida_liw += (maxcol as i64) * (Ns as i64) * m.ida_liw1;

        /* Allocate space for pbar and plist (C `malloc` cannot fail here in
        safe Rust; the C failure branches are unreachable) */
        m.ida_pbar = vec![ZERO; Ns as usize];
        m.ida_plist = vec![0i32; Ns as usize];

        /* Update solver workspace lengths */
        m.ida_lrw += Ns as i64;
        m.ida_liw += Ns as i64;
    }

    SUNTRUE
}

/*
 * IDASensFreeVectors
 *
 * Frees memory allocated by IDASensAllocVectors.
 */

fn IDASensFreeVectors(IDA_mem: &IDAMem) {
    let mut m = IDA_mem.borrow_mut();

    let Ns = m.ida_Ns;

    N_VDestroyVectorArray(std::mem::take(&mut m.ida_deltaS), Ns);
    N_VDestroyVectorArray(std::mem::take(&mut m.ida_ypSpredict), Ns);
    N_VDestroyVectorArray(std::mem::take(&mut m.ida_yySpredict), Ns);
    N_VDestroyVectorArray(std::mem::take(&mut m.ida_ypS), Ns);
    N_VDestroyVectorArray(std::mem::take(&mut m.ida_yyS), Ns);
    N_VDestroyVectorArray(std::mem::take(&mut m.ida_eeS), Ns);
    N_VDestroyVectorArray(std::mem::take(&mut m.ida_ewtS), Ns);
    if let Some(v) = m.ida_tmpS3.take() {
        N_VDestroy(v);
    }

    let maxcol = SUNMAX(m.ida_maxord_alloc, 4);
    for j in 0..=maxcol as usize {
        N_VDestroyVectorArray(std::mem::take(&mut m.ida_phiS[j]), Ns);
    }

    m.ida_pbar = Vec::new();
    m.ida_plist = Vec::new();

    m.ida_lrw -= (((maxcol + 3) as i64) * (Ns as i64) + 1) * m.ida_lrw1 + (Ns as i64);
    m.ida_liw -= (((maxcol + 3) as i64) * (Ns as i64) + 1) * m.ida_liw1 + (Ns as i64);

    if m.ida_VatolSMallocDone {
        N_VDestroyVectorArray(std::mem::take(&mut m.ida_VatolS), Ns);
        m.ida_lrw -= (Ns as i64) * m.ida_lrw1;
        m.ida_liw -= (Ns as i64) * m.ida_liw1;
        m.ida_VatolSMallocDone = SUNFALSE;
    }
    if m.ida_SatolSMallocDone {
        m.ida_SatolS = Vec::new();
        m.ida_lrw -= Ns as i64;
        m.ida_SatolSMallocDone = SUNFALSE;
    }
}

/*
 * IDAQuadSensAllocVectors
 *
 * Create (through duplication) N_Vectors used for quadrature sensitivity analysis,
 * using the N_Vector 'tmpl' as a template.
 */

fn IDAQuadSensAllocVectors(IDA_mem: &IDAMem, tmpl: &N_Vector) -> sunbooleantype {
    let Ns = IDA_mem.borrow().ida_Ns;

    /* Allocate yQS */
    let yyQS = match N_VCloneVectorArray(Ns, tmpl) {
        Some(v) => v,
        None => return SUNFALSE,
    };

    /* Allocate ewtQS */
    let ewtQS = match N_VCloneVectorArray(Ns, tmpl) {
        Some(v) => v,
        None => {
            N_VDestroyVectorArray(yyQS, Ns);
            return SUNFALSE;
        }
    };

    /* Allocate tempvQS */
    let tempvQS = match N_VCloneVectorArray(Ns, tmpl) {
        Some(v) => v,
        None => {
            N_VDestroyVectorArray(yyQS, Ns);
            N_VDestroyVectorArray(ewtQS, Ns);
            return SUNFALSE;
        }
    };

    let eeQS = match N_VCloneVectorArray(Ns, tmpl) {
        Some(v) => v,
        None => {
            N_VDestroyVectorArray(yyQS, Ns);
            N_VDestroyVectorArray(ewtQS, Ns);
            N_VDestroyVectorArray(tempvQS, Ns);
            return SUNFALSE;
        }
    };

    let savrhsQ = match N_VClone(tmpl) {
        Some(v) => v,
        None => {
            N_VDestroyVectorArray(yyQS, Ns);
            N_VDestroyVectorArray(ewtQS, Ns);
            N_VDestroyVectorArray(tempvQS, Ns);
            N_VDestroyVectorArray(eeQS, Ns);
            /* NOTE: upstream omits the `return (SUNFALSE)` here and keeps
            using the arrays it just freed (undefined behavior).  The
            branch is unreachable in this port (N_VClone only fails on an
            allocation failure, which aborts); mapped to the obviously
            intended failure return per ARCHITECTURE deviation class 5. */
            return SUNFALSE;
        }
    };

    {
        let mut m = IDA_mem.borrow_mut();
        m.ida_yyQS = yyQS;
        m.ida_ewtQS = ewtQS;
        m.ida_tempvQS = tempvQS;
        m.ida_eeQS = eeQS;
        m.ida_savrhsQ = Some(savrhsQ);
    }

    let maxcol = SUNMAX(IDA_mem.borrow().ida_maxord, 4);
    /* Allocate phiQS */
    for j in 0..=maxcol as usize {
        match N_VCloneVectorArray(Ns, tmpl) {
            Some(v) => {
                IDA_mem.borrow_mut().ida_phiQS[j] = v;
            }
            None => {
                let mut m = IDA_mem.borrow_mut();
                N_VDestroyVectorArray(std::mem::take(&mut m.ida_yyQS), Ns);
                N_VDestroyVectorArray(std::mem::take(&mut m.ida_ewtQS), Ns);
                N_VDestroyVectorArray(std::mem::take(&mut m.ida_tempvQS), Ns);
                N_VDestroyVectorArray(std::mem::take(&mut m.ida_eeQS), Ns);
                if let Some(v) = m.ida_savrhsQ.take() {
                    N_VDestroy(v);
                }
                for i in 0..j {
                    N_VDestroyVectorArray(std::mem::take(&mut m.ida_phiQS[i]), Ns);
                }
                return SUNFALSE;
            }
        }
    }

    {
        let mut m = IDA_mem.borrow_mut();

        /* Update solver workspace lengths */
        m.ida_lrw += ((maxcol + 5) as i64) * (Ns as i64) * m.ida_lrw1Q;
        m.ida_liw += ((maxcol + 5) as i64) * (Ns as i64) * m.ida_liw1Q;
    }

    SUNTRUE
}

/*
 * IDAQuadSensFreeVectors
 *
 * This routine frees the IDAS vectors allocated in IDAQuadSensAllocVectors.
 */

fn IDAQuadSensFreeVectors(IDA_mem: &IDAMem) {
    let mut m = IDA_mem.borrow_mut();

    let Ns = m.ida_Ns;
    let maxcol = SUNMAX(m.ida_maxord, 4);

    N_VDestroyVectorArray(std::mem::take(&mut m.ida_yyQS), Ns);
    N_VDestroyVectorArray(std::mem::take(&mut m.ida_ewtQS), Ns);
    N_VDestroyVectorArray(std::mem::take(&mut m.ida_eeQS), Ns);
    N_VDestroyVectorArray(std::mem::take(&mut m.ida_tempvQS), Ns);
    if let Some(v) = m.ida_savrhsQ.take() {
        N_VDestroy(v);
    }

    for j in 0..=maxcol as usize {
        N_VDestroyVectorArray(std::mem::take(&mut m.ida_phiQS[j]), Ns);
    }

    m.ida_lrw -= ((maxcol + 5) as i64) * (Ns as i64) * m.ida_lrw1Q;
    m.ida_liw -= ((maxcol + 5) as i64) * (Ns as i64) * m.ida_liw1Q;

    if m.ida_VatolQSMallocDone {
        N_VDestroyVectorArray(std::mem::take(&mut m.ida_VatolQS), Ns);
        m.ida_lrw -= (Ns as i64) * m.ida_lrw1Q;
        m.ida_liw -= (Ns as i64) * m.ida_liw1Q;
    }
    if m.ida_SatolQSMallocDone {
        m.ida_SatolQS = Vec::new();
        m.ida_lrw -= Ns as i64;
    }
    m.ida_VatolQSMallocDone = SUNFALSE;
    m.ida_SatolQSMallocDone = SUNFALSE;
}

/*
 * -----------------------------------------------------------------
 * Initial setup
 * -----------------------------------------------------------------
 */

/*
 * IDAInitialSetup
 *
 * This routine is called by IDASolve once at the first step or when
 * computing consistent initial conditions. It performs all checks
 * on optional inputs and inputs to IDAInit/IDAReInit that could not
 * be done before.
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
    let ier = idab_call_efun(IDA_mem, &phi0, &ewt);
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

    if IDA_mem.borrow().ida_quadr {
        if IDA_mem.borrow().ida_errconQ {
            /* Did the user specify tolerances? */
            if IDA_mem.borrow().ida_itolQ == IDA_NN {
                IDAProcessError(
                    Some(IDA_mem),
                    IDA_ILL_INPUT,
                    line!() as i32,
                    "IDAInitialSetup",
                    file!(),
                    MSG_NO_TOLQ,
                );
                return IDA_ILL_INPUT;
            }

            /* Load ewtQ */
            let (phiQ0, ewtQ) = {
                let m = IDA_mem.borrow();
                (m.ida_phiQ[0].clone().unwrap(), m.ida_ewtQ.clone().unwrap())
            };
            let ier = IDAQuadEwtSet(IDA_mem, &phiQ0, &ewtQ);
            if ier != 0 {
                IDAProcessError(
                    Some(IDA_mem),
                    IDA_ILL_INPUT,
                    line!() as i32,
                    "IDAInitialSetup",
                    file!(),
                    MSG_BAD_EWTQ,
                );
                return IDA_ILL_INPUT;
            }
        }
    } else {
        IDA_mem.borrow_mut().ida_errconQ = SUNFALSE;
    }

    if IDA_mem.borrow().ida_sensi {
        /* Did the user specify tolerances? */
        if IDA_mem.borrow().ida_itolS == IDA_NN {
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

        /* Load ewtS */
        let (phiS0, ewtS) = {
            let m = IDA_mem.borrow();
            (m.ida_phiS[0].clone(), m.ida_ewtS.clone())
        };
        let ier = IDASensEwtSet(IDA_mem, &phiS0, &ewtS);
        if ier != 0 {
            IDAProcessError(
                Some(IDA_mem),
                IDA_ILL_INPUT,
                line!() as i32,
                "IDAInitialSetup",
                file!(),
                MSG_BAD_EWTS,
            );
            return IDA_ILL_INPUT;
        }
    } else {
        IDA_mem.borrow_mut().ida_errconS = SUNFALSE;
    }

    if IDA_mem.borrow().ida_quadr_sensi {
        /* If using the internal DQ functions, we must have access to fQ
         * (i.e. quadrature integration must be enabled) and to the problem parameters */

        if IDA_mem.borrow().ida_rhsQSDQ {
            /* Test if quadratures are defined, so we can use fQ */
            if !IDA_mem.borrow().ida_quadr {
                IDAProcessError(
                    Some(IDA_mem),
                    IDA_ILL_INPUT,
                    line!() as i32,
                    "IDAInitialSetup",
                    file!(),
                    MSG_NULL_RHSQ,
                );
                return IDA_ILL_INPUT;
            }

            /* Test if we have the problem parameters */
            if IDA_mem.borrow().ida_p.is_none() {
                IDAProcessError(
                    Some(IDA_mem),
                    IDA_ILL_INPUT,
                    line!() as i32,
                    "IDAInitialSetup",
                    file!(),
                    MSG_NULL_P,
                );
                return IDA_ILL_INPUT;
            }
        }

        if IDA_mem.borrow().ida_errconQS {
            /* Did the user specify tolerances? */
            if IDA_mem.borrow().ida_itolQS == IDA_NN {
                IDAProcessError(
                    Some(IDA_mem),
                    IDA_ILL_INPUT,
                    line!() as i32,
                    "IDAInitialSetup",
                    file!(),
                    MSG_NO_TOLQS,
                );
                return IDA_ILL_INPUT;
            }

            /* If needed, did the user provide quadrature tolerances? */
            let (itolQS, itolQ) = {
                let m = IDA_mem.borrow();
                (m.ida_itolQS, m.ida_itolQ)
            };
            if (itolQS == IDA_EE) && (itolQ == IDA_NN) {
                IDAProcessError(
                    Some(IDA_mem),
                    IDA_ILL_INPUT,
                    line!() as i32,
                    "IDAInitialSetup",
                    file!(),
                    MSG_NO_TOLQ,
                );
                return IDA_ILL_INPUT;
            }

            /* Load ewtS */
            let (phiQS0, ewtQS) = {
                let m = IDA_mem.borrow();
                (m.ida_phiQS[0].clone(), m.ida_ewtQS.clone())
            };
            let ier = IDAQuadSensEwtSet(IDA_mem, &phiQS0, &ewtQS);
            if ier != 0 {
                IDAProcessError(
                    Some(IDA_mem),
                    IDA_ILL_INPUT,
                    line!() as i32,
                    "IDAInitialSetup",
                    file!(),
                    MSG_BAD_EWTQS,
                );
                return IDA_ILL_INPUT;
            }
        }
    } else {
        IDA_mem.borrow_mut().ida_errconQS = SUNFALSE;
    }

    /* Check to see if y0 satisfies constraints. */
    let constraints = IDA_mem.borrow().ida_constraints.clone();
    if let Some(constraints) = constraints {
        let (sensi, ism) = {
            let m = IDA_mem.borrow();
            (m.ida_sensi, m.ida_ism)
        };
        if sensi && (ism == IDA_SIMULTANEOUS) {
            IDAProcessError(
                Some(IDA_mem),
                IDA_ILL_INPUT,
                line!() as i32,
                "IDAInitialSetup",
                file!(),
                MSG_BAD_ISM_CONSTR,
            );
            return IDA_ILL_INPUT;
        }

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

    /* always initialize the DAE NLS in case the user disables sensitivities later */
    let ier = crate::idas_nls::idaNlsInit(IDA_mem);
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

    if IDA_mem.borrow().NLSsim.is_some() {
        let ier = crate::idas_nls_sim::idaNlsInitSensSim(IDA_mem);
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
    }

    if IDA_mem.borrow().NLSstg.is_some() {
        let ier = crate::idas_nls_stg::idaNlsInitSensStg(IDA_mem);
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
    }

    IDA_SUCCESS
}

/*
 * IDAQuadSetup
 *
 * This routine is called by IDASolve once at the first step. It
 * fills in phiQ[1] and phiQS[1] since they are not provided by
 * the user. It is important this is NOT done in IDAInitialSetup as
 * IDACalcIC will call IDAInitialSetup in which case inconsistent initial
 * conditions will be used to compute phiQ[1] and phiQS[1] (if
 * IDAQuadInit is called BEFORE IDACalcIC) or phiQ[1] and phiQS[1]
 * will not be initialized (if IDAQuadInit is called AFTER IDACalcIC).
 */
fn IDAQuadSetup(IDA_mem: &IDAMem) -> i32 {
    if IDA_mem.borrow().ida_quadr {
        /* Evaluate quadrature rhs and set phiQ[1] */
        let (tn, phi0, phi1, phiQ1) = {
            let m = IDA_mem.borrow();
            (
                m.ida_tn,
                m.ida_phi[0].clone().unwrap(),
                m.ida_phi[1].clone().unwrap(),
                m.ida_phiQ[1].clone().unwrap(),
            )
        };
        let ier = idab_call_rhsQ(IDA_mem, tn, &phi0, &phi1, &phiQ1);
        IDA_mem.borrow_mut().ida_nrQe += 1;
        if ier < 0 {
            /* NOTE: upstream passes the MSG_QRHSFUNC_FAILED format string
            without its `t` argument here (`idas.c:5181` — a missing-vararg
            defect, so release C prints an indeterminate value for the `%g`);
            this port supplies `ida_tn`, the value every sibling call site
            uses. NOT deviation class 5 (that covers C UB -> deterministic
            PANIC); this is the missing-vararg substitution class — see
            ARCHITECTURE "Accepted deviation classes". Reachable only when
            the quadrature RHS fails unrecoverably on the very first step,
            which no reference example does. */
            let tn = IDA_mem.borrow().ida_tn;
            IDAProcessError(
                Some(IDA_mem),
                IDA_QRHS_FAIL,
                line!() as i32,
                "IDAQuadSetup",
                file!(),
                &MSG_QRHSFUNC_FAILED(tn),
            );
            return IDA_QRHS_FAIL;
        } else if ier > 0 {
            IDAProcessError(
                Some(IDA_mem),
                IDA_FIRST_QRHS_ERR,
                line!() as i32,
                "IDAQuadSetup",
                file!(),
                MSG_QRHSFUNC_FIRST,
            );
            return IDA_FIRST_QRHS_ERR;
        }
    }

    if IDA_mem.borrow().ida_quadr_sensi {
        /* store the quadrature sensitivity residual. */
        let (Ns, tn, phi0, phi1, phiS0, phiS1, phiQ1, phiQS1, tmpS1, tmpS2, tmpS3) = {
            let m = IDA_mem.borrow();
            (
                m.ida_Ns,
                m.ida_tn,
                m.ida_phi[0].clone().unwrap(),
                m.ida_phi[1].clone().unwrap(),
                m.ida_phiS[0].clone(),
                m.ida_phiS[1].clone(),
                m.ida_phiQ[1].clone().unwrap(),
                m.ida_phiQS[1].clone(),
                m.ida_tmpS1.clone().unwrap(),
                m.ida_tmpS2.clone().unwrap(),
                m.ida_tmpS3.clone().unwrap(),
            )
        };
        let ier = idab_call_rhsQS(
            IDA_mem, Ns, tn, &phi0, &phi1, &phiS0, &phiS1, &phiQ1, &phiQS1, &tmpS1, &tmpS2, &tmpS3,
        );
        IDA_mem.borrow_mut().ida_nrQSe += 1;
        if ier < 0 {
            /* NOTE: same missing-vararg defect as above for
            MSG_QSRHSFUNC_FAILED (`idas.c:5205`); `ida_tn` supplied, same
            deviation class. */
            let tn = IDA_mem.borrow().ida_tn;
            IDAProcessError(
                Some(IDA_mem),
                IDA_QSRHS_FAIL,
                line!() as i32,
                "IDAQuadSetup",
                file!(),
                &MSG_QSRHSFUNC_FAILED(tn),
            );
            /* NOTE: upstream reports IDA_QSRHS_FAIL but returns
            IDA_QRHS_FAIL — preserved verbatim. */
            return IDA_QRHS_FAIL;
        } else if ier > 0 {
            IDAProcessError(
                Some(IDA_mem),
                IDA_FIRST_QSRHS_ERR,
                line!() as i32,
                "IDAQuadSetup",
                file!(),
                MSG_QSRHSFUNC_FIRST,
            );
            return IDA_FIRST_QSRHS_ERR;
        }
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
 * IDAQuadEwtSet
 *
 */

fn IDAQuadEwtSet(IDA_mem: &IDAMem, qcur: &N_Vector, weightQ: &N_Vector) -> i32 {
    let itolQ = IDA_mem.borrow().ida_itolQ;
    let flag: i32 = match itolQ {
        IDA_SS => IDAQuadEwtSetSS(IDA_mem, qcur, weightQ),
        IDA_SV => IDAQuadEwtSetSV(IDA_mem, qcur, weightQ),
        _ => 0,
    };

    flag
}

/*
 * IDAQuadEwtSetSS
 *
 */

fn IDAQuadEwtSetSS(IDA_mem: &IDAMem, qcur: &N_Vector, weightQ: &N_Vector) -> i32 {
    /* Use ypQ as temporary storage */
    let (tempvQ, rtolQ, SatolQ, atolQmin0) = {
        let m = IDA_mem.borrow();
        (
            m.ida_ypQ.clone().unwrap(),
            m.ida_rtolQ,
            m.ida_SatolQ,
            m.ida_atolQmin0,
        )
    };

    N_VAbs(qcur, &tempvQ);
    N_VScale(rtolQ, &tempvQ, &tempvQ);
    N_VAddConst(&tempvQ, SatolQ, &tempvQ);
    if atolQmin0 {
        if N_VMin(&tempvQ) <= ZERO {
            return -1;
        }
    }
    N_VInv(&tempvQ, weightQ);

    0
}

/*
 * IDAQuadEwtSetSV
 *
 */

fn IDAQuadEwtSetSV(IDA_mem: &IDAMem, qcur: &N_Vector, weightQ: &N_Vector) -> i32 {
    /* Use ypQ as temporary storage */
    let (tempvQ, rtolQ, VatolQ, atolQmin0) = {
        let m = IDA_mem.borrow();
        (
            m.ida_ypQ.clone().unwrap(),
            m.ida_rtolQ,
            m.ida_VatolQ.clone().unwrap(),
            m.ida_atolQmin0,
        )
    };

    N_VAbs(qcur, &tempvQ);
    N_VLinearSum(rtolQ, &tempvQ, ONE, &VatolQ, &tempvQ);
    if atolQmin0 {
        if N_VMin(&tempvQ) <= ZERO {
            return -1;
        }
    }
    N_VInv(&tempvQ, weightQ);

    0
}

/*
 * IDASensEwtSet
 *
 */

pub fn IDASensEwtSet(IDA_mem: &IDAMem, yScur: &[N_Vector], weightS: &[N_Vector]) -> i32 {
    let itolS = IDA_mem.borrow().ida_itolS;
    let flag: i32 = match itolS {
        IDA_EE => IDASensEwtSetEE(IDA_mem, yScur, weightS),
        IDA_SS => IDASensEwtSetSS(IDA_mem, yScur, weightS),
        IDA_SV => IDASensEwtSetSV(IDA_mem, yScur, weightS),
        _ => 0,
    };

    flag
}

/*
 * IDASensEwtSetEE
 *
 * In this case, the error weight vector for the i-th sensitivity is set to
 *
 * ewtS_i = pbar_i * efun(pbar_i*yS_i)
 *
 * In other words, the scaled sensitivity pbar_i * yS_i has the same error
 * weight vector calculation as the solution vector.
 *
 */

fn IDASensEwtSetEE(IDA_mem: &IDAMem, yScur: &[N_Vector], weightS: &[N_Vector]) -> i32 {
    /* Use tempv1 as temporary storage for the scaled sensitivity */
    let (pyS, Ns) = {
        let m = IDA_mem.borrow();
        (m.ida_tempv1.clone().unwrap(), m.ida_Ns)
    };

    for is in 0..Ns as usize {
        let pbar_is = IDA_mem.borrow().ida_pbar[is];
        N_VScale(pbar_is, &yScur[is], &pyS);
        let flag = idab_call_efun(IDA_mem, &pyS, &weightS[is]);
        if flag != 0 {
            return -1;
        }
        N_VScale(pbar_is, &weightS[is], &weightS[is]);
    }

    0
}

/*
 * IDASensEwtSetSS
 *
 */

fn IDASensEwtSetSS(IDA_mem: &IDAMem, yScur: &[N_Vector], weightS: &[N_Vector]) -> i32 {
    let (tempv1, rtolS, Ns) = {
        let m = IDA_mem.borrow();
        (m.ida_tempv1.clone().unwrap(), m.ida_rtolS, m.ida_Ns)
    };

    for is in 0..Ns as usize {
        let (SatolS_is, atolSmin0_is) = {
            let m = IDA_mem.borrow();
            (m.ida_SatolS[is], m.ida_atolSmin0[is])
        };
        N_VAbs(&yScur[is], &tempv1);
        N_VScale(rtolS, &tempv1, &tempv1);
        N_VAddConst(&tempv1, SatolS_is, &tempv1);
        if atolSmin0_is {
            if N_VMin(&tempv1) <= ZERO {
                return -1;
            }
        }
        N_VInv(&tempv1, &weightS[is]);
    }
    0
}

/*
 * IDASensEwtSetSV
 *
 */

fn IDASensEwtSetSV(IDA_mem: &IDAMem, yScur: &[N_Vector], weightS: &[N_Vector]) -> i32 {
    let (tempv1, rtolS, Ns) = {
        let m = IDA_mem.borrow();
        (m.ida_tempv1.clone().unwrap(), m.ida_rtolS, m.ida_Ns)
    };

    for is in 0..Ns as usize {
        let (VatolS_is, atolSmin0_is) = {
            let m = IDA_mem.borrow();
            (m.ida_VatolS[is].clone(), m.ida_atolSmin0[is])
        };
        N_VAbs(&yScur[is], &tempv1);
        N_VLinearSum(rtolS, &tempv1, ONE, &VatolS_is, &tempv1);
        if atolSmin0_is {
            if N_VMin(&tempv1) <= ZERO {
                return -1;
            }
        }
        N_VInv(&tempv1, &weightS[is]);
    }

    0
}

/*
 * IDAQuadSensEwtSet
 *
 */

pub fn IDAQuadSensEwtSet(IDA_mem: &IDAMem, yQScur: &[N_Vector], weightQS: &[N_Vector]) -> i32 {
    let itolQS = IDA_mem.borrow().ida_itolQS;
    let flag: i32 = match itolQS {
        IDA_EE => IDAQuadSensEwtSetEE(IDA_mem, yQScur, weightQS),
        IDA_SS => IDAQuadSensEwtSetSS(IDA_mem, yQScur, weightQS),
        IDA_SV => IDAQuadSensEwtSetSV(IDA_mem, yQScur, weightQS),
        _ => 0,
    };

    flag
}

/*
 * IDAQuadSensEwtSetEE
 *
 * In this case, the error weight vector for the i-th quadrature sensitivity
 * is set to
 *
 * ewtQS_i = pbar_i * IDAQuadEwtSet(pbar_i*yQS_i)
 *
 * In other words, the scaled sensitivity pbar_i * yQS_i has the same error
 * weight vector calculation as the quadrature vector.
 *
 */
fn IDAQuadSensEwtSetEE(IDA_mem: &IDAMem, yQScur: &[N_Vector], weightQS: &[N_Vector]) -> i32 {
    /* Use tempvQS[0] as temporary storage for the scaled sensitivity */
    let (pyS, Ns) = {
        let m = IDA_mem.borrow();
        (m.ida_tempvQS[0].clone(), m.ida_Ns)
    };

    for is in 0..Ns as usize {
        let pbar_is = IDA_mem.borrow().ida_pbar[is];
        N_VScale(pbar_is, &yQScur[is], &pyS);
        let flag = IDAQuadEwtSet(IDA_mem, &pyS, &weightQS[is]);
        if flag != 0 {
            return -1;
        }
        N_VScale(pbar_is, &weightQS[is], &weightQS[is]);
    }

    0
}

fn IDAQuadSensEwtSetSS(IDA_mem: &IDAMem, yQScur: &[N_Vector], weightQS: &[N_Vector]) -> i32 {
    /* Use ypQ as temporary storage */
    let (tempvQ, rtolQS, Ns) = {
        let m = IDA_mem.borrow();
        (m.ida_ypQ.clone().unwrap(), m.ida_rtolQS, m.ida_Ns)
    };

    for is in 0..Ns as usize {
        let (SatolQS_is, atolQSmin0_is) = {
            let m = IDA_mem.borrow();
            (m.ida_SatolQS[is], m.ida_atolQSmin0[is])
        };
        N_VAbs(&yQScur[is], &tempvQ);
        N_VScale(rtolQS, &tempvQ, &tempvQ);
        N_VAddConst(&tempvQ, SatolQS_is, &tempvQ);
        if atolQSmin0_is {
            if N_VMin(&tempvQ) <= ZERO {
                return -1;
            }
        }
        N_VInv(&tempvQ, &weightQS[is]);
    }

    0
}

fn IDAQuadSensEwtSetSV(IDA_mem: &IDAMem, yQScur: &[N_Vector], weightQS: &[N_Vector]) -> i32 {
    /* Use ypQ as temporary storage */
    let (tempvQ, rtolQS, Ns) = {
        let m = IDA_mem.borrow();
        (m.ida_ypQ.clone().unwrap(), m.ida_rtolQS, m.ida_Ns)
    };

    for is in 0..Ns as usize {
        let (VatolQS_is, atolQSmin0_is) = {
            let m = IDA_mem.borrow();
            (m.ida_VatolQS[is].clone(), m.ida_atolQSmin0[is])
        };
        N_VAbs(&yQScur[is], &tempvQ);
        N_VLinearSum(rtolQS, &tempvQ, ONE, &VatolQS_is, &tempvQ);
        if atolQSmin0_is {
            if N_VMin(&tempvQ) <= ZERO {
                return -1;
            }
        }
        N_VInv(&tempvQ, &weightQS[is]);
    }

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
            let past = {
                let m = IDA_mem.borrow();
                (m.ida_tn - m.ida_tstop) * m.ida_hh > ZERO
            };
            if past {
                let (tstop, tn) = {
                    let m = IDA_mem.borrow();
                    (m.ida_tstop, m.ida_tn)
                };
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
                let _ier = IDAGetSolution(IDA_mem, tn, yret, ypret);
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
                IDAGetSolution(IDA_mem, tstop, yret, ypret);
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
                IDAGetSolution(IDA_mem, tout, yret, ypret);
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
            IDA_ERR_FAIL
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
            IDA_CONV_FAIL
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
            IDA_LSETUP_FAIL
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
            IDA_LSOLVE_FAIL
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
            IDA_REP_RES_ERR
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
            IDA_RES_FAIL
        }

        IDA_REP_QRHS_ERR => {
            IDAProcessError(
                Some(IDA_mem),
                IDA_REP_QRHS_ERR,
                line!() as i32,
                "IDAHandleFailure",
                file!(),
                &MSG_QRHSFUNC_REPTD(tn),
            );
            IDA_REP_QRHS_ERR
        }

        IDA_QRHS_FAIL => {
            IDAProcessError(
                Some(IDA_mem),
                IDA_QRHS_FAIL,
                line!() as i32,
                "IDAHandleFailure",
                file!(),
                &MSG_QRHSFUNC_FAILED(tn),
            );
            IDA_QRHS_FAIL
        }

        IDA_REP_SRES_ERR => {
            IDAProcessError(
                Some(IDA_mem),
                IDA_REP_SRES_ERR,
                line!() as i32,
                "IDAHandleFailure",
                file!(),
                &MSG_SRES_REPTD(tn),
            );
            IDA_REP_SRES_ERR
        }

        IDA_SRES_FAIL => {
            IDAProcessError(
                Some(IDA_mem),
                IDA_SRES_FAIL,
                line!() as i32,
                "IDAHandleFailure",
                file!(),
                &MSG_SRES_FAILED(tn),
            );
            IDA_SRES_FAIL
        }

        IDA_REP_QSRHS_ERR => {
            IDAProcessError(
                Some(IDA_mem),
                IDA_REP_QSRHS_ERR,
                line!() as i32,
                "IDAHandleFailure",
                file!(),
                &MSG_QSRHSFUNC_REPTD(tn),
            );
            IDA_REP_QSRHS_ERR
        }

        IDA_QSRHS_FAIL => {
            IDAProcessError(
                Some(IDA_mem),
                IDA_QSRHS_FAIL,
                line!() as i32,
                "IDAHandleFailure",
                file!(),
                &MSG_QSRHSFUNC_FAILED(tn),
            );
            IDA_QSRHS_FAIL
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
            IDA_CONSTR_FAIL
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
            IDA_MEM_NULL
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
            IDA_MEM_NULL
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
            IDA_NLS_SETUP_FAIL
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
            IDA_NLS_FAIL
        }

        _ => {
            /* This return should never happen */
            IDAProcessError(
                Some(IDA_mem),
                IDA_UNRECOGNIZED_ERROR,
                line!() as i32,
                "IDAHandleFailure",
                file!(),
                "IDA encountered an unrecognized error. Please report this \
                 to the Sundials developers at sundials-users@llnl.gov",
            );
            IDA_UNRECOGNIZED_ERROR
        }
    }
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
    let mut ck: sunrealtype = ZERO;

    /* Are we computing sensitivities with the staggered or simultaneous approach? */
    let (sensi_stg, sensi_sim) = {
        let m = IDA_mem.borrow();
        (
            m.ida_sensi && (m.ida_ism == IDA_STAGGERED),
            m.ida_sensi && (m.ida_ism == IDA_SIMULTANEOUS),
        )
    };

    let saved_t = IDA_mem.borrow().ida_tn;

    /* Initialize failure counters for this step attempt */

    let mut ncf: i32 = 0; /* corrector failures  */
    let mut nef: i32 = 0; /* error test failures */
    let mut step_constraint_fails: i32 = 0;

    if IDA_mem.borrow().ida_nst == 0 {
        let mut m = IDA_mem.borrow_mut();
        let hh = m.ida_hh;
        m.ida_kk = 1;
        m.ida_kused = 0;
        m.ida_hused = ZERO;
        m.ida_psi[0] = hh;
        m.ida_cj = ONE / hh;
        m.ida_phase = 0;
        m.ida_ns = 0;
    }

    /* To prevent 'unintialized variable' warnings */
    let mut err_k = ZERO;
    let mut err_km1 = ZERO;
    let mut err_km2 = ZERO;

    /* Looping point for attempts to take a step */

    loop {
        /*-----------------------
        Set method coefficients
        -----------------------*/

        IDASetCoeffs(IDA_mem, &mut ck);

        /* C assigns `kflag = IDA_SUCCESS` here; the value is overwritten
        before every read, so `kflag` is bound per-branch below. */

        /*----------------------------------------------------
        If tn is past tstop (by roundoff), reset it to tstop.
        -----------------------------------------------------*/

        {
            let mut m = IDA_mem.borrow_mut();
            let hh = m.ida_hh;
            let tn = m.ida_tn;
            m.ida_tn = tn + hh;
            if m.ida_tstopset {
                let tstop = m.ida_tstop;
                if (m.ida_tn - tstop) * hh > ZERO {
                    m.ida_tn = tstop;
                }
            }
        }

        /*-----------------------
        Advance state variables
        -----------------------*/

        /* Compute predicted values for yy and yp */
        IDAPredict(IDA_mem);

        /* Compute predicted values for yyS and ypS (if simultaneous approach) */
        if sensi_sim {
            let (yySpredict, ypSpredict) = {
                let m = IDA_mem.borrow();
                (m.ida_yySpredict.clone(), m.ida_ypSpredict.clone())
            };
            IDASensPredict(IDA_mem, &yySpredict, &ypSpredict);
        }

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
            nflag = IDATestError(IDA_mem, ck, &mut err_k, &mut err_km1, &mut err_km2);
        }

        /* Test for convergence or error test failures */
        if nflag != IDA_SUCCESS {
            /* restore and decide what to do */
            IDARestore(IDA_mem, saved_t);

            /* C passes the addresses of ida_ncfn / ida_netf; mirror the
            write-back so the mem fields stay in sync. */
            let kflag = {
                let (mut ncfn, mut netf) = {
                    let m = IDA_mem.borrow();
                    (m.ida_ncfn, m.ida_netf)
                };
                let kflag = IDAHandleNFlag(
                    IDA_mem, nflag, err_k, err_km1, &mut ncfn, &mut ncf, &mut netf, &mut nef,
                );
                {
                    let mut m = IDA_mem.borrow_mut();
                    m.ida_ncfn = ncfn;
                    m.ida_netf = netf;
                }
                kflag
            };

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

        /*----------------------------
        Advance quadrature variables
        ----------------------------*/
        if IDA_mem.borrow().ida_quadr {
            nflag = IDAQuadNls(IDA_mem);

            /* If NLS was successful, perform error test */
            if IDA_mem.borrow().ida_errconQ && (nflag == IDA_SUCCESS) {
                nflag = IDAQuadTestError(IDA_mem, ck, &mut err_k, &mut err_km1, &mut err_km2);
            }

            /* Test for convergence or error test failures */
            if nflag != IDA_SUCCESS {
                /* restore and decide what to do */
                IDARestore(IDA_mem, saved_t);
                let kflag = {
                    let (mut ncfnQ, mut netfQ) = {
                        let m = IDA_mem.borrow();
                        (m.ida_ncfnQ, m.ida_netfQ)
                    };
                    let kflag = IDAHandleNFlag(
                        IDA_mem, nflag, err_k, err_km1, &mut ncfnQ, &mut ncf, &mut netfQ, &mut nef,
                    );
                    {
                        let mut m = IDA_mem.borrow_mut();
                        m.ida_ncfnQ = ncfnQ;
                        m.ida_netfQ = netfQ;
                    }
                    kflag
                };

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
        }

        /*--------------------------------------------------
        Advance sensitivity variables (Staggered approach)
        --------------------------------------------------*/
        if sensi_stg {
            /* Evaluate res at converged y, needed for future evaluations of sens. RHS
            If res() fails recoverably, treat it as a convergence failure and
            attempt the step again */

            let (tn, yy, yp, delta) = {
                let m = IDA_mem.borrow();
                (
                    m.ida_tn,
                    m.ida_yy.clone().unwrap(),
                    m.ida_yp.clone().unwrap(),
                    m.ida_delta.clone().unwrap(),
                )
            };
            let retval = idab_call_res(IDA_mem, tn, &yy, &yp, &delta);

            if retval < 0 {
                return IDA_RES_FAIL;
            }
            if retval > 0 {
                continue;
            }

            /* Compute predicted values for yyS and ypS */
            let (yySpredict, ypSpredict) = {
                let m = IDA_mem.borrow();
                (m.ida_yySpredict.clone(), m.ida_ypSpredict.clone())
            };
            IDASensPredict(IDA_mem, &yySpredict, &ypSpredict);

            /* Nonlinear system solution */
            nflag = IDASensNls(IDA_mem);

            /* If NLS was successful, perform error test */
            if IDA_mem.borrow().ida_errconS && (nflag == IDA_SUCCESS) {
                nflag = IDASensTestError(IDA_mem, ck, &mut err_k, &mut err_km1, &mut err_km2);
            }

            /* Test for convergence or error test failures */
            if nflag != IDA_SUCCESS {
                /* restore and decide what to do */
                IDARestore(IDA_mem, saved_t);
                /* NOTE: upstream passes the QUADRATURE counters (ncfnQ /
                netfQ) here, not the sensitivity ones — preserved verbatim. */
                let kflag = {
                    let (mut ncfnQ, mut netfQ) = {
                        let m = IDA_mem.borrow();
                        (m.ida_ncfnQ, m.ida_netfQ)
                    };
                    let kflag = IDAHandleNFlag(
                        IDA_mem, nflag, err_k, err_km1, &mut ncfnQ, &mut ncf, &mut netfQ, &mut nef,
                    );
                    {
                        let mut m = IDA_mem.borrow_mut();
                        m.ida_ncfnQ = ncfnQ;
                        m.ida_netfQ = netfQ;
                    }
                    kflag
                };

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
        }

        /*-------------------------------------------
        Advance quadrature sensitivity variables
        -------------------------------------------*/
        if IDA_mem.borrow().ida_quadr_sensi {
            nflag = IDAQuadSensNls(IDA_mem);

            /* If NLS was successful, perform error test */
            if IDA_mem.borrow().ida_errconQS && (nflag == IDA_SUCCESS) {
                nflag = IDAQuadSensTestError(IDA_mem, ck, &mut err_k, &mut err_km1, &mut err_km2);
            }

            /* Test for convergence or error test failures */
            if nflag != IDA_SUCCESS {
                /* restore and decide what to do */
                IDARestore(IDA_mem, saved_t);
                let kflag = {
                    let (mut ncfnQ, mut netfQ) = {
                        let m = IDA_mem.borrow();
                        (m.ida_ncfnQ, m.ida_netfQ)
                    };
                    let kflag = IDAHandleNFlag(
                        IDA_mem, nflag, err_k, err_km1, &mut ncfnQ, &mut ncf, &mut netfQ, &mut nef,
                    );
                    {
                        let mut m = IDA_mem.borrow_mut();
                        m.ida_ncfnQ = ncfnQ;
                        m.ida_netfQ = netfQ;
                    }
                    kflag
                };

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
/* =================================================================
 * idas.c PART C -- fragment of the `idas` module.
 *
 * Covers every function of `src/idas/idas.c` whose definition begins
 * at line 6000 or later:
 *
 *   Step coefficients      : IDASetCoeffs
 *   Nonlinear solver       : IDANls, IDACheckConstraints, IDAPredict,
 *                            IDAQuadNls, IDAQuadPredict, IDASensNls,
 *                            IDASensPredict, IDAQuadSensNls,
 *                            IDAQuadSensPredict
 *   Error test             : IDATestError, IDAQuadTestError,
 *                            IDASensTestError, IDAQuadSensTestError,
 *                            IDARestore
 *   Failure handling       : IDAHandleNFlag, IDAReset
 *   After a good step      : IDACompleteStep
 *   Interpolated output    : IDAGetSolution
 *   Norms                  : IDAWrmsNorm, IDASensWrmsNorm,
 *                            IDAQuadSensWrmsNorm, IDAQuadWrmsNormUpdate,
 *                            IDASensWrmsNormUpdate,
 *                            IDAQuadSensWrmsNormUpdate
 *   Rootfinding            : IDARcheck1, IDARcheck2, IDARcheck3,
 *                            IDARootfind
 *   Internal sensitivity DQ: IDASensResDQ, IDASensRes1DQ,
 *                            IDAQuadSensRhsInternalDQ,
 *                            IDAQuadSensRhs1InternalDQ
 *
 * (`IDAProcessError`, defined at idas.c:8875, is relocated to
 * `idas_impl.rs` per the frozen contract and is NOT redefined here.)
 *
 * Fragment protocol (identical to `ida__part_b.rs`): no `use` items and
 * no module-scope consts -- the concatenation target `idas.rs` supplies
 * them (`use crate::idas_impl::*;` plus the `sundials_core` imports);
 * anything exotic is spelled with a fully-qualified `sundials_core::...`
 * / `std::...` path.  Every module-scope `#define` of idas.c (ZERO,
 * HALF, ONE, TWO, TWENTY, HUNDRED, PT9, PT1, PT0001, ONEPSM,
 * PREDICT_AGAIN, UNSET/LOWER/RAISE/MAINTAIN, ERROR_TEST_FAIL,
 * RTFOUND/CLOSERT, CENTERED1/2, FORWARD1/2, ...) lives in
 * `idas_impl.rs` and reaches this fragment through that glob import.
 *
 * Reference build: SUNDIALS_LOGGING_LEVEL = 2 (SUNLogInfo/SUNLogInfoIf/
 * SUNLogDebug/SUNLogExtraDebug* call sites omitted at translation time;
 * IDA_WARNING paths kept -- they queue through the logger), profiling
 * off, error checks off (SUNAssert/SUNCheck* no-ops), monitoring on,
 * serial branches only.
 *
 * Borrow discipline: internal functions take `&IDAMem` and use granular
 * borrows -- no borrow of the mem is ever held across a user callback,
 * an N_Vector operation, an `IDAProcessError` call, or a linear/
 * nonlinear-solver call, all of which can re-enter the mem.
 * =================================================================*/

/* -----------------------------------------------------------------
 * FRAGMENT-LOCAL CALLBACK INVOCATION HELPERS
 *
 * Named `idac_*` so this fragment never collides with identically
 * shaped helpers in a sibling `idas.c` fragment; the integrator may
 * dedupe them into a single set when concatenating.
 *
 * Binding invariant D: the `Option<Box<dyn Any>>` data token is taken
 * out of the mem around every user-callback call and restored on EVERY
 * path (there are no early returns inside these helpers).
 * -----------------------------------------------------------------*/

/// Invoke the user residual `res`
/// (C: `IDA_mem->ida_res(t, yy, yp, rr, IDA_mem->ida_user_data)`).
fn idac_call_res(
    IDA_mem: &IDAMem,
    tt: sunrealtype,
    yy: &N_Vector,
    yp: &N_Vector,
    rr: &N_Vector,
) -> i32 {
    let res = IDA_mem.borrow().ida_res.expect("ida_res set");
    let mut user_data = IDA_mem.borrow_mut().ida_user_data.take();
    let retval = res(tt, yy, yp, rr, &mut user_data);
    IDA_mem.borrow_mut().ida_user_data = user_data;
    retval
}

/// Invoke the quadrature right-hand side `rhsQ`
/// (C: `IDA_mem->ida_rhsQ(t, yy, yp, rrQ, IDA_mem->ida_user_data)`).
fn idac_call_rhsQ(
    IDA_mem: &IDAMem,
    tres: sunrealtype,
    yy: &N_Vector,
    yp: &N_Vector,
    rrQ: &N_Vector,
) -> i32 {
    let rhsQ = IDA_mem.borrow().ida_rhsQ.expect("ida_rhsQ set");
    let mut user_data = IDA_mem.borrow_mut().ida_user_data.take();
    let retval = rhsQ(tres, yy, yp, rrQ, &mut user_data);
    IDA_mem.borrow_mut().ida_user_data = user_data;
    retval
}

/// Invoke the quadrature-sensitivity RHS `rhsQS` (C:
/// `IDA_mem->ida_rhsQS(Ns, t, yy, yp, yyS, ypS, rrQ, rhsvalQS,
/// IDA_mem->ida_user_dataQS, yytmp, yptmp, tmpQS)`).
///
/// `ida_user_dataQS` is `Some(token)` when IDAS uses its internal DQ
/// routine (C stored `IDA_mem` there) and `None` when C stored
/// `ida_user_data`; the `None` case therefore forwards the integrator's
/// `ida_user_data` box (same sentinel convention as the sibling
/// fragment's `idab_call_rhsQS`).
#[allow(clippy::too_many_arguments)]
fn idac_call_rhsQS(
    IDA_mem: &IDAMem,
    Ns: i32,
    t: sunrealtype,
    yy: &N_Vector,
    yp: &N_Vector,
    yyS: &[N_Vector],
    ypS: &[N_Vector],
    rrQ: &N_Vector,
    rhsvalQS: &[N_Vector],
    yytmp: &N_Vector,
    yptmp: &N_Vector,
    tmpQS: &N_Vector,
) -> i32 {
    let rhsQS = IDA_mem.borrow().ida_rhsQS.expect("ida_rhsQS set");
    let mut token = IDA_mem.borrow_mut().ida_user_dataQS.take();
    if token.is_some() {
        let retval = rhsQS(
            Ns, t, yy, yp, yyS, ypS, rrQ, rhsvalQS, &mut token, yytmp, yptmp, tmpQS,
        );
        IDA_mem.borrow_mut().ida_user_dataQS = token;
        retval
    } else {
        let mut user_data = IDA_mem.borrow_mut().ida_user_data.take();
        let retval = rhsQS(
            Ns,
            t,
            yy,
            yp,
            yyS,
            ypS,
            rrQ,
            rhsvalQS,
            &mut user_data,
            yytmp,
            yptmp,
            tmpQS,
        );
        let mut m = IDA_mem.borrow_mut();
        m.ida_user_data = user_data;
        m.ida_user_dataQS = token;
        retval
    }
}

/// Invoke the user root function `g`
/// (C: `IDA_mem->ida_gfun(t, yy, yp, gout, IDA_mem->ida_user_data)`).
fn idac_call_gfun(
    IDA_mem: &IDAMem,
    t: sunrealtype,
    y: &N_Vector,
    yp: &N_Vector,
    gout: &mut [sunrealtype],
) -> i32 {
    let gfun = IDA_mem.borrow().ida_gfun.expect("ida_gfun set");
    let mut user_data = IDA_mem.borrow_mut().ida_user_data.take();
    let retval = gfun(t, y, yp, gout, &mut user_data);
    IDA_mem.borrow_mut().ida_user_data = user_data;
    retval
}

/// Mirror the C writes `IDA_mem->ida_Xvecs[j] = v` (j = 0 .. src.len()-1)
/// into a scratch handle array WITHOUT shrinking it: C writes the first
/// `j` slots of a `malloc`ed array that stays `MXORDP1` (or
/// `Ns*MXORDP1`) long, so the port keeps any surplus tail entries and
/// only grows when the C array would already have been long enough.
fn idac_store_scratch(dst: &mut Vec<N_Vector>, src: &[N_Vector]) {
    for (j, v) in src.iter().enumerate() {
        if j < dst.len() {
            dst[j] = v.clone();
        } else {
            dst.push(v.clone());
        }
    }
}
/* ---------------- END FRAGMENT-LOCAL HELPERS ---------------------- */

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

        let (quadr, sensi, quadr_sensi, Ns) = {
            let m = IDA_mem.borrow();
            (m.ida_quadr, m.ida_sensi, m.ida_quadr_sensi, m.ida_Ns)
        };

        let (cvals, phivecs) = {
            let mut m = IDA_mem.borrow_mut();
            for i in ns..=kk {
                m.ida_cvals[i - ns] = m.ida_beta[i];
            }
            let cvals: Vec<sunrealtype> = m.ida_cvals[0..=(kk - ns)].to_vec();
            let phivecs: Vec<N_Vector> = (ns..=kk).map(|j| m.ida_phi[j].clone().unwrap()).collect();
            (cvals, phivecs)
        };

        /* C passes the same `ida_phi + ns` pointer for X and Z; the port passes
        the same slice so the array-pointer equality test in the vector op
        selects the identical in-place branch. */
        let _ = N_VScaleVectorArray((kk - ns + 1) as i32, &cvals, &phivecs, &phivecs);

        if quadr {
            let phiQvecs: Vec<N_Vector> = {
                let m = IDA_mem.borrow();
                (ns..=kk).map(|j| m.ida_phiQ[j].clone().unwrap()).collect()
            };
            let _ = N_VScaleVectorArray((kk - ns + 1) as i32, &cvals, &phiQvecs, &phiQvecs);
        }

        if sensi || quadr_sensi {
            let mut m = IDA_mem.borrow_mut();
            let mut j = 0usize;
            for i in ns..=kk {
                for _is in 0..Ns as usize {
                    m.ida_cvals[j] = m.ida_beta[i];
                    j += 1;
                }
            }
        }

        if sensi {
            let (j, cvalsS, Xvecs) = {
                let mut m = IDA_mem.borrow_mut();
                let mut Xvecs: Vec<N_Vector> = Vec::new();
                for i in ns..=kk {
                    for is in 0..Ns as usize {
                        Xvecs.push(m.ida_phiS[i][is].clone());
                    }
                }
                let j = Xvecs.len();
                let cvalsS: Vec<sunrealtype> = m.ida_cvals[0..j].to_vec();
                idac_store_scratch(&mut m.ida_Xvecs, &Xvecs);
                (j, cvalsS, Xvecs)
            };

            let _ = N_VScaleVectorArray(j as i32, &cvalsS, &Xvecs, &Xvecs);
        }

        if quadr_sensi {
            let (j, cvalsS, Xvecs) = {
                let mut m = IDA_mem.borrow_mut();
                let mut Xvecs: Vec<N_Vector> = Vec::new();
                for i in ns..=kk {
                    for is in 0..Ns as usize {
                        Xvecs.push(m.ida_phiQS[i][is].clone());
                    }
                }
                let j = Xvecs.len();
                let cvalsS: Vec<sunrealtype> = m.ida_cvals[0..j].to_vec();
                idac_store_scratch(&mut m.ida_Xvecs, &Xvecs);
                (j, cvalsS, Xvecs)
            };

            let _ = N_VScaleVectorArray(j as i32, &cvalsS, &Xvecs, &Xvecs);
        }
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
 *  IDA_SRES_RECVR      IDA_SRES_FAIL
 *  IDA_LSETUP_RECVR    IDA_LSETUP_FAIL
 *  IDA_LSOLVE_RECVR    IDA_LSOLVE_FAIL
 *
 *  SUN_NLS_CONV_RECVR
 *  IDA_MEM_NULL
 */

fn IDANls(IDA_mem: &IDAMem) -> i32 {
    let mut nni_inc: i64 = 0;
    let mut nnf_inc: i64 = 0;

    /* Are we computing sensitivities with the IDA_SIMULTANEOUS approach? */
    let sensi_sim = {
        let m = IDA_mem.borrow();
        m.ida_sensi && (m.ida_ism == IDA_SIMULTANEOUS)
    };

    let mut callLSetup: sunbooleantype = SUNFALSE;

    {
        let mut m = IDA_mem.borrow_mut();

        /* Initialize if the first time called */

        if m.ida_nst == 0 {
            m.ida_cjold = m.ida_cj;
            m.ida_ss = TWENTY;
            m.ida_ssS = TWENTY;
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
            if m.ida_forceSetup {
                callLSetup = SUNTRUE;
            }
            if m.ida_cj != m.ida_cjlast {
                m.ida_ss = HUNDRED;
                m.ida_ssS = HUNDRED;
            }
        }
    }

    /* initial guess for the correction to the predictor */
    let (ee, ycorSim) = {
        let m = IDA_mem.borrow();
        (m.ida_ee.clone().unwrap(), m.ycorSim.clone())
    };
    if sensi_sim {
        N_VConst(ZERO, ycorSim.as_ref().unwrap());
    } else {
        N_VConst(ZERO, &ee);
    }

    /* The C `void* IDA_mem` handed to the nonlinear solver maps to a boxed
    handle clone (the same token shape idas_nls*.rs downcasts). */
    let NLS = IDA_mem.borrow().NLS.clone().unwrap();
    let mut nls_mem: Option<Box<dyn std::any::Any>> = Some(Box::new(IDA_mem.clone()));

    /* call nonlinear solver setup if it exists */
    if NLS.ops.borrow().setup.is_some() {
        let retval = if sensi_sim {
            SUNNonlinSolSetup(&NLS, ycorSim.as_ref().unwrap(), &mut nls_mem)
        } else {
            SUNNonlinSolSetup(&NLS, &ee, &mut nls_mem)
        };

        if retval < 0 {
            return IDA_NLS_SETUP_FAIL;
        }
        if retval > 0 {
            return IDA_NLS_SETUP_RECVR;
        }
    }

    /* solve the nonlinear system */
    let retval;
    if sensi_sim {
        let (NLSsim, ypredictSim, ewtSim, epsNewt) = {
            let m = IDA_mem.borrow();
            (
                m.NLSsim.clone().unwrap(),
                m.ypredictSim.clone().unwrap(),
                m.ewtSim.clone().unwrap(),
                m.ida_epsNewt,
            )
        };
        retval = SUNNonlinSolSolve(
            &NLSsim,
            &ypredictSim,
            ycorSim.as_ref().unwrap(),
            &ewtSim,
            epsNewt,
            callLSetup,
            &mut nls_mem,
        );

        /* increment counters */
        let _ = SUNNonlinSolGetNumIters(&NLSsim, &mut nni_inc);
        IDA_mem.borrow_mut().ida_nni += nni_inc;

        let _ = SUNNonlinSolGetNumConvFails(&NLSsim, &mut nnf_inc);
        IDA_mem.borrow_mut().ida_nnf += nnf_inc;
    } else {
        let (yypredict, ewt, epsNewt) = {
            let m = IDA_mem.borrow();
            (
                m.ida_yypredict.clone().unwrap(),
                m.ida_ewt.clone().unwrap(),
                m.ida_epsNewt,
            )
        };
        retval = SUNNonlinSolSolve(
            &NLS,
            &yypredict,
            &ee,
            &ewt,
            epsNewt,
            callLSetup,
            &mut nls_mem,
        );

        /* increment counter */
        let _ = SUNNonlinSolGetNumIters(&NLS, &mut nni_inc);
        IDA_mem.borrow_mut().ida_nni += nni_inc;

        let _ = SUNNonlinSolGetNumConvFails(&NLS, &mut nnf_inc);
        IDA_mem.borrow_mut().ida_nnf += nnf_inc;
    }

    /* return if nonlinear solver failed */
    if retval != sundials_core::sundials_errors::SUN_SUCCESS {
        return retval;
    }

    /* update yy and yp based on the final correction from the nonlinear solver */
    let (yypredict, yppredict, cj, yy, yp) = {
        let m = IDA_mem.borrow();
        (
            m.ida_yypredict.clone().unwrap(),
            m.ida_yppredict.clone().unwrap(),
            m.ida_cj,
            m.ida_yy.clone().unwrap(),
            m.ida_yp.clone().unwrap(),
        )
    };
    N_VLinearSum(ONE, &yypredict, ONE, &ee, &yy);
    N_VLinearSum(ONE, &yppredict, cj, &ee, &yp);

    /* update the sensitivities based on the final correction from the nonlinear solver */
    if sensi_sim {
        let (Ns, yySpredict, ypSpredict, eeS, yyS, ypS) = {
            let m = IDA_mem.borrow();
            (
                m.ida_Ns,
                m.ida_yySpredict.clone(),
                m.ida_ypSpredict.clone(),
                m.ida_eeS.clone(),
                m.ida_yyS.clone(),
                m.ida_ypS.clone(),
            )
        };
        let _ = N_VLinearSumVectorArray(Ns, ONE, &yySpredict, ONE, &eeS, &yyS);
        let _ = N_VLinearSumVectorArray(Ns, ONE, &ypSpredict, cj, &eeS, &ypS);
    }

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
 * IDAQuadNls
 *
 * This routine solves for the quadrature variables at the new step.
 * It does not solve a nonlinear system, but rather updates the
 * quadrature variables. The name for this function is just for
 * uniformity purposes.
 *
 */

fn IDAQuadNls(IDA_mem: &IDAMem) -> i32 {
    /* Predict: load yyQ and ypQ */
    IDAQuadPredict(IDA_mem);

    /* Compute correction eeQ */
    let (tn, yy, yp, eeQ) = {
        let m = IDA_mem.borrow();
        (
            m.ida_tn,
            m.ida_yy.clone().unwrap(),
            m.ida_yp.clone().unwrap(),
            m.ida_eeQ.clone().unwrap(),
        )
    };
    let retval = idac_call_rhsQ(IDA_mem, tn, &yy, &yp, &eeQ);
    IDA_mem.borrow_mut().ida_nrQe += 1;
    if retval < 0 {
        return IDA_QRHS_FAIL;
    } else if retval > 0 {
        return IDA_QRHS_RECVR;
    }

    if IDA_mem.borrow().ida_quadr_sensi {
        let savrhsQ = IDA_mem.borrow().ida_savrhsQ.clone().unwrap();
        N_VScale(ONE, &eeQ, &savrhsQ);
    }

    let (ypQ, cj, yyQ) = {
        let m = IDA_mem.borrow();
        (
            m.ida_ypQ.clone().unwrap(),
            m.ida_cj,
            m.ida_yyQ.clone().unwrap(),
        )
    };
    N_VLinearSum(ONE, &eeQ, -ONE, &ypQ, &eeQ);
    N_VScale(ONE / cj, &eeQ, &eeQ);

    /* Apply correction: yyQ = yyQ + eeQ */
    N_VLinearSum(ONE, &yyQ, ONE, &eeQ, &yyQ);

    IDA_SUCCESS
}

/*
 * IDAQuadPredict
 *
 * This routine predicts the new value for vectors yyQ and ypQ
 */

fn IDAQuadPredict(IDA_mem: &IDAMem) {
    let (kk, cvals, gvals, phiQvecs, phiQvecs1, yyQ, ypQ) = {
        let mut m = IDA_mem.borrow_mut();
        let kk = m.ida_kk as usize;

        for j in 0..=kk {
            m.ida_cvals[j] = ONE;
        }

        let cvals: Vec<sunrealtype> = m.ida_cvals[0..=kk].to_vec();
        /* C: `ida_gamma + 1` / `ida_phiQ + 1` pointer offsets */
        let gvals: Vec<sunrealtype> = m.ida_gamma[1..=kk].to_vec();
        let phiQvecs: Vec<N_Vector> = (0..=kk).map(|j| m.ida_phiQ[j].clone().unwrap()).collect();
        let phiQvecs1: Vec<N_Vector> = (1..=kk).map(|j| m.ida_phiQ[j].clone().unwrap()).collect();

        (
            kk,
            cvals,
            gvals,
            phiQvecs,
            phiQvecs1,
            m.ida_yyQ.clone().unwrap(),
            m.ida_ypQ.clone().unwrap(),
        )
    };

    let _ = N_VLinearCombination((kk + 1) as i32, &cvals, &phiQvecs, &yyQ);

    let _ = N_VLinearCombination(kk as i32, &gvals, &phiQvecs1, &ypQ);
}

/*
 * IDASensNls
 *
 * This routine attempts to solve, one by one, all the sensitivity
 * linear systems using nonlinear iterations and the linear solver
 * specified (Staggered approach).
 */

fn IDASensNls(IDA_mem: &IDAMem) -> i32 {
    let mut nniS_inc: i64 = 0;
    let mut nnfS_inc: i64 = 0;

    let callLSetup: sunbooleantype = SUNFALSE;

    /* initial guess for the correction to the predictor */
    let ycorStg = IDA_mem.borrow().ycorStg.clone().unwrap();
    N_VConst(ZERO, &ycorStg);

    /* solve the nonlinear system */
    let (NLSstg, ypredictStg, ewtStg, epsNewt) = {
        let m = IDA_mem.borrow();
        (
            m.NLSstg.clone().unwrap(),
            m.ypredictStg.clone().unwrap(),
            m.ewtStg.clone().unwrap(),
            m.ida_epsNewt,
        )
    };
    /* The C `void* IDA_mem` handed to the nonlinear solver maps to a boxed
    handle clone (the same token shape idas_nls_stg.rs downcasts). */
    let mut nls_mem: Option<Box<dyn std::any::Any>> = Some(Box::new(IDA_mem.clone()));
    let retval = SUNNonlinSolSolve(
        &NLSstg,
        &ypredictStg,
        &ycorStg,
        &ewtStg,
        epsNewt,
        callLSetup,
        &mut nls_mem,
    );

    /* increment counters */
    let _ = SUNNonlinSolGetNumIters(&NLSstg, &mut nniS_inc);
    IDA_mem.borrow_mut().ida_nniS += nniS_inc;

    let _ = SUNNonlinSolGetNumConvFails(&NLSstg, &mut nnfS_inc);
    IDA_mem.borrow_mut().ida_nnfS += nnfS_inc;

    if retval != sundials_core::sundials_errors::SUN_SUCCESS {
        IDA_mem.borrow_mut().ida_ncfnS += 1;
        return retval;
    }

    /* update using the final correction from the nonlinear solver */
    let (Ns, yySpredict, ypSpredict, cj, eeS, yyS, ypS) = {
        let m = IDA_mem.borrow();
        (
            m.ida_Ns,
            m.ida_yySpredict.clone(),
            m.ida_ypSpredict.clone(),
            m.ida_cj,
            m.ida_eeS.clone(),
            m.ida_yyS.clone(),
            m.ida_ypS.clone(),
        )
    };
    let _ = N_VLinearSumVectorArray(Ns, ONE, &yySpredict, ONE, &eeS, &yyS);
    let _ = N_VLinearSumVectorArray(Ns, ONE, &ypSpredict, cj, &eeS, &ypS);

    retval
}

/*
 * IDASensPredict
 *
 * This routine loads the predicted values for the is-th sensitivity
 * in the vectors yySens and ypSens.
 *
 * When ism=IDA_STAGGERED,  yySens = yyS[is] and ypSens = ypS[is]
 */

fn IDASensPredict(IDA_mem: &IDAMem, yySens: &[N_Vector], ypSens: &[N_Vector]) {
    let (Ns, kk, cvals, gvals, phiS, phiS1) = {
        let mut m = IDA_mem.borrow_mut();
        let kk = m.ida_kk as usize;

        for j in 0..=kk {
            m.ida_cvals[j] = ONE;
        }

        let cvals: Vec<sunrealtype> = m.ida_cvals[0..=kk].to_vec();
        /* C: `ida_gamma + 1` / `ida_phiS + 1` pointer offsets */
        let gvals: Vec<sunrealtype> = m.ida_gamma[1..=kk].to_vec();
        let phiS: Vec<Vec<N_Vector>> = m.ida_phiS[0..=kk].to_vec();
        let phiS1: Vec<Vec<N_Vector>> = m.ida_phiS[1..=kk].to_vec();

        (m.ida_Ns, kk, cvals, gvals, phiS, phiS1)
    };

    let _ = N_VLinearCombinationVectorArray(Ns, (kk + 1) as i32, &cvals, &phiS, yySens);

    let _ = N_VLinearCombinationVectorArray(Ns, kk as i32, &gvals, &phiS1, ypSens);
}

/*
 * IDAQuadSensNls
 *
 * This routine solves for the snesitivity quadrature variables at the
 * new step. It does not solve a nonlinear system, but rather updates
 * the sensitivity variables. The name for this function is just for
 * uniformity purposes.
 *
 */

fn IDAQuadSensNls(IDA_mem: &IDAMem) -> i32 {
    /* Predict: load yyQS and ypQS for each sensitivity. Store
    1st order information in tempvQS. */

    let (yyQS, ypQS) = {
        let m = IDA_mem.borrow();
        (m.ida_yyQS.clone(), m.ida_tempvQS.clone())
    };
    IDAQuadSensPredict(IDA_mem, &yyQS, &ypQS);

    /* Compute correction eeQS */
    let (Ns, tn, yy, yp, yyS, ypS, savrhsQ, eeQS, tmpS1, tmpS2, tmpS3) = {
        let m = IDA_mem.borrow();
        (
            m.ida_Ns,
            m.ida_tn,
            m.ida_yy.clone().unwrap(),
            m.ida_yp.clone().unwrap(),
            m.ida_yyS.clone(),
            m.ida_ypS.clone(),
            m.ida_savrhsQ.clone().unwrap(),
            m.ida_eeQS.clone(),
            m.ida_tmpS1.clone().unwrap(),
            m.ida_tmpS2.clone().unwrap(),
            m.ida_tmpS3.clone().unwrap(),
        )
    };
    let retval = idac_call_rhsQS(
        IDA_mem, Ns, tn, &yy, &yp, &yyS, &ypS, &savrhsQ, &eeQS, &tmpS1, &tmpS2, &tmpS3,
    );
    IDA_mem.borrow_mut().ida_nrQSe += 1;

    if retval < 0 {
        return IDA_QSRHS_FAIL;
    } else if retval > 0 {
        return IDA_QSRHS_RECVR;
    }

    let cj = IDA_mem.borrow().ida_cj;
    let retval = N_VLinearSumVectorArray(Ns, ONE / cj, &eeQS, -ONE / cj, &ypQS, &eeQS);
    if retval != IDA_SUCCESS {
        return IDA_VECTOROP_ERR;
    }

    /* Apply correction: yyQS[is] = yyQ[is] + eeQ[is] */
    let retval = N_VLinearSumVectorArray(Ns, ONE, &yyQS, ONE, &eeQS, &yyQS);
    if retval != IDA_SUCCESS {
        return IDA_VECTOROP_ERR;
    }

    IDA_SUCCESS
}

/*
 * IDAQuadSensPredict
 *
 * This routine predicts the new value for vectors yyQS and ypQS
 */

fn IDAQuadSensPredict(IDA_mem: &IDAMem, yQS: &[N_Vector], ypQS: &[N_Vector]) {
    let (Ns, kk, cvals, gvals, phiQS, phiQS1) = {
        let mut m = IDA_mem.borrow_mut();
        let kk = m.ida_kk as usize;

        for j in 0..=kk {
            m.ida_cvals[j] = ONE;
        }

        let cvals: Vec<sunrealtype> = m.ida_cvals[0..=kk].to_vec();
        /* C: `ida_gamma + 1` / `ida_phiQS + 1` pointer offsets */
        let gvals: Vec<sunrealtype> = m.ida_gamma[1..=kk].to_vec();
        let phiQS: Vec<Vec<N_Vector>> = m.ida_phiQS[0..=kk].to_vec();
        let phiQS1: Vec<Vec<N_Vector>> = m.ida_phiQS[1..=kk].to_vec();

        (m.ida_Ns, kk, cvals, gvals, phiQS, phiQS1)
    };

    let _ = N_VLinearCombinationVectorArray(Ns, (kk + 1) as i32, &cvals, &phiQS, yQS);

    let _ = N_VLinearCombinationVectorArray(Ns, kk as i32, &gvals, &phiQS1, ypQS);
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
    err_km2: &mut sunrealtype,
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
            *err_km2 = IDA_mem.borrow().ida_sigma[(kk - 2) as usize] * enorm_km2;
            let terr_km2 = (kk - 1) as sunrealtype * (*err_km2);

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
 * IDAQuadTestError
 *
 * This routine estimates quadrature errors and updates errors at
 * orders k, k-1, k-2, decides whether or not to suggest an order reduction,
 * and performs the local error test.
 *
 * IDAQuadTestError returns the updated local error estimate at orders k,
 * k-1, and k-2. These are norms of type SUNMAX(|err|,|errQ|).
 *
 * The return flag can be either IDA_SUCCESS or ERROR_TEST_FAIL.
 */

fn IDAQuadTestError(
    IDA_mem: &IDAMem,
    ck: sunrealtype,
    err_k: &mut sunrealtype,
    err_km1: &mut sunrealtype,
    err_km2: &mut sunrealtype,
) -> i32 {
    let mut check_for_reduction: sunbooleantype = SUNFALSE;

    /* Rename ypQ */
    let (tempv, eeQ, ewtQ, kk) = {
        let m = IDA_mem.borrow();
        (
            m.ida_ypQ.clone().unwrap(),
            m.ida_eeQ.clone().unwrap(),
            m.ida_ewtQ.clone().unwrap(),
            m.ida_kk,
        )
    };

    /* Update error for order k. */
    let enormQ = N_VWrmsNorm(&eeQ, &ewtQ);
    let errQ_k = IDA_mem.borrow().ida_sigma[kk as usize] * enormQ;
    if errQ_k > *err_k {
        *err_k = errQ_k;
        check_for_reduction = SUNTRUE;
    }
    let terr_k = (kk + 1) as sunrealtype * (*err_k);

    if kk > 1 {
        /* Update error at order k-1 */
        let phiQ_kk = IDA_mem.borrow().ida_phiQ[kk as usize].clone().unwrap();
        N_VLinearSum(ONE, &phiQ_kk, ONE, &eeQ, &tempv);
        /* (port) the norm is evaluated into a local first: no mem borrow may
        be held across a vector operation.  Operand order of the product is
        irrelevant to the result. */
        let nrm_km1 = N_VWrmsNorm(&tempv, &ewtQ);
        let errQ_km1 = IDA_mem.borrow().ida_sigma[(kk - 1) as usize] * nrm_km1;
        if errQ_km1 > *err_km1 {
            *err_km1 = errQ_km1;
            check_for_reduction = SUNTRUE;
        }
        let terr_km1 = kk as sunrealtype * (*err_km1);

        /* Has an order decrease already been decided in IDATestError? */
        if IDA_mem.borrow().ida_knew != kk {
            check_for_reduction = SUNFALSE;
        }

        if check_for_reduction {
            if kk > 2 {
                /* Update error at order k-2 */
                let phiQ_km1 = IDA_mem.borrow().ida_phiQ[(kk - 1) as usize]
                    .clone()
                    .unwrap();
                N_VLinearSum(ONE, &phiQ_km1, ONE, &tempv, &tempv);
                let nrm_km2 = N_VWrmsNorm(&tempv, &ewtQ);
                let errQ_km2 = IDA_mem.borrow().ida_sigma[(kk - 2) as usize] * nrm_km2;
                if errQ_km2 > *err_km2 {
                    *err_km2 = errQ_km2;
                }
                let terr_km2 = (kk - 1) as sunrealtype * (*err_km2);

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
    }

    /* Perform error test */
    if ck * enormQ > ONE {
        ERROR_TEST_FAIL
    } else {
        IDA_SUCCESS
    }
}

/*
 * IDASensTestError
 *
 * This routine estimates sensitivity errors and updates errors at
 * orders k, k-1, k-2, decides whether or not to suggest an order reduction,
 * and performs the local error test. (Used only in staggered approach).
 *
 * IDASensTestError returns the updated local error estimate at orders k,
 * k-1, and k-2. These are norms of type SUNMAX(|err|,|errQ|,|errS|).
 *
 * The return flag can be either IDA_SUCCESS or ERROR_TEST_FAIL.
 */

fn IDASensTestError(
    IDA_mem: &IDAMem,
    ck: sunrealtype,
    err_k: &mut sunrealtype,
    err_km1: &mut sunrealtype,
    err_km2: &mut sunrealtype,
) -> i32 {
    let mut check_for_reduction: sunbooleantype = SUNFALSE;

    /* Rename deltaS */
    let (tempv, Ns, eeS, ewtS, suppressalg, kk) = {
        let m = IDA_mem.borrow();
        (
            m.ida_deltaS.clone(),
            m.ida_Ns,
            m.ida_eeS.clone(),
            m.ida_ewtS.clone(),
            m.ida_suppressalg,
            m.ida_kk,
        )
    };

    /* Update error for order k. */
    let enormS = IDASensWrmsNorm(IDA_mem, &eeS, &ewtS, suppressalg);
    let errS_k = IDA_mem.borrow().ida_sigma[kk as usize] * enormS;
    if errS_k > *err_k {
        *err_k = errS_k;
        check_for_reduction = SUNTRUE;
    }
    let terr_k = (kk + 1) as sunrealtype * (*err_k);

    if kk > 1 {
        /* Update error at order k-1 */
        let phiS_kk = IDA_mem.borrow().ida_phiS[kk as usize].clone();
        let retval = N_VLinearSumVectorArray(Ns, ONE, &phiS_kk, ONE, &eeS, &tempv);
        if retval != IDA_SUCCESS {
            return IDA_VECTOROP_ERR;
        }

        /* (port) IDASensWrmsNorm re-borrows the mem (it uses `ida_cvals` as
        scratch), so the norm is evaluated into a local before the `ida_sigma`
        read.  Operand order of the product is irrelevant to the result. */
        let nrmS_km1 = IDASensWrmsNorm(IDA_mem, &tempv, &ewtS, suppressalg);
        let errS_km1 = IDA_mem.borrow().ida_sigma[(kk - 1) as usize] * nrmS_km1;

        if errS_km1 > *err_km1 {
            *err_km1 = errS_km1;
            check_for_reduction = SUNTRUE;
        }
        let terr_km1 = kk as sunrealtype * (*err_km1);

        /* Has an order decrease already been decided in IDATestError? */
        if IDA_mem.borrow().ida_knew != kk {
            check_for_reduction = SUNFALSE;
        }

        if check_for_reduction {
            if kk > 2 {
                /* Update error at order k-2 */
                let phiS_km1 = IDA_mem.borrow().ida_phiS[(kk - 1) as usize].clone();
                let retval = N_VLinearSumVectorArray(Ns, ONE, &phiS_km1, ONE, &tempv, &tempv);
                if retval != IDA_SUCCESS {
                    return IDA_VECTOROP_ERR;
                }

                let nrmS_km2 = IDASensWrmsNorm(IDA_mem, &tempv, &ewtS, suppressalg);
                let errS_km2 = IDA_mem.borrow().ida_sigma[(kk - 2) as usize] * nrmS_km2;

                if errS_km2 > *err_km2 {
                    *err_km2 = errS_km2;
                }
                let terr_km2 = (kk - 1) as sunrealtype * (*err_km2);

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
    }

    /* Perform error test */
    if ck * enormS > ONE {
        ERROR_TEST_FAIL
    } else {
        IDA_SUCCESS
    }
}

/*
 * IDAQuadSensTestError
 *
 * This routine estimates quadrature sensitivity errors and updates
 * errors at orders k, k-1, k-2, decides whether or not to suggest
 * an order reduction and performs the local error test. (Used
 * only in staggered approach).
 *
 * IDAQuadSensTestError returns the updated local error estimate at
 * orders k, k-1, and k-2. These are norms of type
 * SUNMAX(|err|,|errQ|,|errS|,|errQS|).
 *
 * The return flag can be either IDA_SUCCESS or ERROR_TEST_FAIL.
 */

fn IDAQuadSensTestError(
    IDA_mem: &IDAMem,
    ck: sunrealtype,
    err_k: &mut sunrealtype,
    err_km1: &mut sunrealtype,
    err_km2: &mut sunrealtype,
) -> i32 {
    let mut check_for_reduction: sunbooleantype = SUNFALSE;

    let (tempv, Ns, eeQS, ewtQS, kk) = {
        let m = IDA_mem.borrow();
        (
            m.ida_yyQS.clone(),
            m.ida_Ns,
            m.ida_eeQS.clone(),
            m.ida_ewtQS.clone(),
            m.ida_kk,
        )
    };

    let enormQS = IDAQuadSensWrmsNorm(IDA_mem, &eeQS, &ewtQS);
    let errQS_k = IDA_mem.borrow().ida_sigma[kk as usize] * enormQS;

    if errQS_k > *err_k {
        *err_k = errQS_k;
        check_for_reduction = SUNTRUE;
    }
    let terr_k = (kk + 1) as sunrealtype * (*err_k);

    if kk > 1 {
        /* Update error at order k-1 */
        let phiQS_kk = IDA_mem.borrow().ida_phiQS[kk as usize].clone();
        let retval = N_VLinearSumVectorArray(Ns, ONE, &phiQS_kk, ONE, &eeQS, &tempv);
        if retval != IDA_SUCCESS {
            return IDA_VECTOROP_ERR;
        }

        /* (port) IDAQuadSensWrmsNorm re-borrows the mem (`ida_cvals` scratch),
        so the norm is evaluated into a local before the `ida_sigma` read. */
        let nrmQS_km1 = IDAQuadSensWrmsNorm(IDA_mem, &tempv, &ewtQS);
        let errQS_km1 = IDA_mem.borrow().ida_sigma[(kk - 1) as usize] * nrmQS_km1;

        if errQS_km1 > *err_km1 {
            *err_km1 = errQS_km1;
            check_for_reduction = SUNTRUE;
        }
        let terr_km1 = kk as sunrealtype * (*err_km1);

        /* Has an order decrease already been decided in IDATestError? */
        if IDA_mem.borrow().ida_knew != kk {
            check_for_reduction = SUNFALSE;
        }

        if check_for_reduction {
            if kk > 2 {
                /* Update error at order k-2 */
                let phiQS_km1 = IDA_mem.borrow().ida_phiQS[(kk - 1) as usize].clone();
                let retval = N_VLinearSumVectorArray(Ns, ONE, &phiQS_km1, ONE, &tempv, &tempv);
                if retval != IDA_SUCCESS {
                    return IDA_VECTOROP_ERR;
                }

                let nrmQS_km2 = IDAQuadSensWrmsNorm(IDA_mem, &tempv, &ewtQS);
                let errQS_km2 = IDA_mem.borrow().ida_sigma[(kk - 2) as usize] * nrmQS_km2;

                if errQS_km2 > *err_km2 {
                    *err_km2 = errQS_km2;
                }
                let terr_km2 = (kk - 1) as sunrealtype * (*err_km2);

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
    }

    /* Perform error test */
    if ck * enormQS > ONE {
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
        for i in 1..=kk {
            m.ida_psi[i - 1] = m.ida_psi[i] - m.ida_hh;
        }

        (m.ida_ns, m.ida_kk)
    };

    if ns <= kk {
        let ns = ns as usize;
        let kk = kk as usize;

        let (quadr, sensi, quadr_sensi, Ns) = {
            let m = IDA_mem.borrow();
            (m.ida_quadr, m.ida_sensi, m.ida_quadr_sensi, m.ida_Ns)
        };

        let (cvals, phivecs) = {
            let mut m = IDA_mem.borrow_mut();
            for i in ns..=kk {
                m.ida_cvals[i - ns] = ONE / m.ida_beta[i];
            }
            let cvals: Vec<sunrealtype> = m.ida_cvals[0..=(kk - ns)].to_vec();
            let phivecs: Vec<N_Vector> = (ns..=kk).map(|j| m.ida_phi[j].clone().unwrap()).collect();
            (cvals, phivecs)
        };

        /* Same slice for X and Z: selects the in-place branch, as in C where
        both arguments are the `ida_phi + ns` pointer. */
        let _ = N_VScaleVectorArray((kk - ns + 1) as i32, &cvals, &phivecs, &phivecs);

        if quadr {
            let phiQvecs: Vec<N_Vector> = {
                let m = IDA_mem.borrow();
                (ns..=kk).map(|j| m.ida_phiQ[j].clone().unwrap()).collect()
            };
            let _ = N_VScaleVectorArray((kk - ns + 1) as i32, &cvals, &phiQvecs, &phiQvecs);
        }

        if sensi || quadr_sensi {
            let mut m = IDA_mem.borrow_mut();
            let mut j = 0usize;
            for i in ns..=kk {
                for _is in 0..Ns as usize {
                    m.ida_cvals[j] = ONE / m.ida_beta[i];
                    j += 1;
                }
            }
        }

        if sensi {
            let (j, cvalsS, Xvecs) = {
                let mut m = IDA_mem.borrow_mut();
                let mut Xvecs: Vec<N_Vector> = Vec::new();
                for i in ns..=kk {
                    for is in 0..Ns as usize {
                        Xvecs.push(m.ida_phiS[i][is].clone());
                    }
                }
                let j = Xvecs.len();
                let cvalsS: Vec<sunrealtype> = m.ida_cvals[0..j].to_vec();
                idac_store_scratch(&mut m.ida_Xvecs, &Xvecs);
                (j, cvalsS, Xvecs)
            };

            let _ = N_VScaleVectorArray(j as i32, &cvalsS, &Xvecs, &Xvecs);
        }

        if quadr_sensi {
            let (j, cvalsS, Xvecs) = {
                let mut m = IDA_mem.borrow_mut();
                let mut Xvecs: Vec<N_Vector> = Vec::new();
                for i in ns..=kk {
                    for is in 0..Ns as usize {
                        Xvecs.push(m.ida_phiQS[i][is].clone());
                    }
                }
                let j = Xvecs.len();
                let cvalsS: Vec<sunrealtype> = m.ida_cvals[0..j].to_vec();
                idac_store_scratch(&mut m.ida_Xvecs, &Xvecs);
                (j, cvalsS, Xvecs)
            };

            let _ = N_VScaleVectorArray(j as i32, &cvalsS, &Xvecs, &Xvecs);
        }
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
 *   IDA_CONSTR_RECVR           > 0
 *   SUN_NLS_CONV_RECVR         > 0
 *   IDA_QRHS_RECVR             > 0
 *   IDA_QSRHS_RECVR            > 0
 *   IDA_RES_FAIL               < 0
 *   IDA_LSOLVE_FAIL            < 0
 *   IDA_LSETUP_FAIL            < 0
 *   IDA_QRHS_FAIL              < 0
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
 *   IDA_QRHS_FAIL
 *   IDA_REP_QRHS_ERR
 *
 * NOTE (port): C passes `&IDA_mem->ida_ncfn` / `&IDA_mem->ida_netf` (or
 * the `...Q` variants) as `ncfnPtr` / `netfPtr` — a `&mut` into those
 * RefCell fields cannot coexist with the `&IDAMem` argument, so the
 * caller (IDAStep) copies the selected mem counter into a local, passes
 * `&mut` that local, and writes it straight back (binding invariant B).
 * Nothing in this routine reads those two counters, so the copy-in /
 * copy-out is exactly equivalent to C's pointer channel.
 */

fn IDAHandleNFlag(
    IDA_mem: &IDAMem,
    nflag: i32,
    err_k: sunrealtype,
    err_km1: sunrealtype,
    ncfnPtr: &mut i64,
    ncfPtr: &mut i32,
    netfPtr: &mut i64,
    nefPtr: &mut i32,
) -> i32 {
    IDA_mem.borrow_mut().ida_phase = 1;

    if nflag != ERROR_TEST_FAIL {
        /*-----------------------
        Nonlinear solver failed
        -----------------------*/

        *ncfPtr += 1; /* local counter for convergence failures */
        *ncfnPtr += 1; /* global counter for convergence failures */

        if nflag < 0 {
            /* nonrecoverable failure */

            if nflag == IDA_LSOLVE_FAIL {
                IDA_LSOLVE_FAIL
            } else if nflag == IDA_LSETUP_FAIL {
                IDA_LSETUP_FAIL
            } else if nflag == IDA_RES_FAIL {
                IDA_RES_FAIL
            } else if nflag == IDA_QRHS_FAIL {
                IDA_QRHS_FAIL
            } else if nflag == IDA_SRES_FAIL {
                IDA_SRES_FAIL
            } else if nflag == IDA_QSRHS_FAIL {
                IDA_QSRHS_FAIL
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
                    if nflag == IDA_QRHS_RECVR {
                        return IDA_REP_QRHS_ERR;
                    }
                    if nflag == IDA_SRES_RECVR {
                        return IDA_REP_SRES_ERR;
                    }
                    if nflag == IDA_QSRHS_RECVR {
                        return IDA_REP_QSRHS_ERR;
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
        *netfPtr += 1; /* global counter for error test failures */

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
    let (eta, phi1, quadr, sensi, quadr_sensi, Ns) = {
        let mut m = IDA_mem.borrow_mut();
        m.ida_psi[0] = m.ida_hh;
        (
            m.ida_eta,
            m.ida_phi[1].clone().unwrap(),
            m.ida_quadr,
            m.ida_sensi,
            m.ida_quadr_sensi,
            m.ida_Ns,
        )
    };

    N_VScale(eta, &phi1, &phi1);

    if quadr {
        let phiQ1 = IDA_mem.borrow().ida_phiQ[1].clone().unwrap();
        N_VScale(eta, &phiQ1, &phiQ1);
    }

    if sensi || quadr_sensi {
        let mut m = IDA_mem.borrow_mut();
        for is in 0..Ns as usize {
            m.ida_cvals[is] = eta;
        }
    }

    if sensi {
        let (cvals, phiS1) = {
            let m = IDA_mem.borrow();
            (m.ida_cvals[0..Ns as usize].to_vec(), m.ida_phiS[1].clone())
        };
        let _ = N_VScaleVectorArray(Ns, &cvals, &phiS1, &phiS1);
    }

    if quadr_sensi {
        let (cvals, phiQS1) = {
            let m = IDA_mem.borrow();
            (m.ida_cvals[0..Ns as usize].to_vec(), m.ida_phiQS[1].clone())
        };
        let _ = N_VScaleVectorArray(Ns, &cvals, &phiQS1, &phiQS1);
    }
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
            let mut enorm = IDAWrmsNorm(IDA_mem, &tempv1, &ewt, suppressalg);

            if IDA_mem.borrow().ida_errconQ {
                /* Rename ypQ */
                let (tempvQ, eeQ, phiQ_kp1, ewtQ) = {
                    let m = IDA_mem.borrow();
                    (
                        m.ida_ypQ.clone().unwrap(),
                        m.ida_eeQ.clone().unwrap(),
                        m.ida_phiQ[(kk + 1) as usize].clone().unwrap(),
                        m.ida_ewtQ.clone().unwrap(),
                    )
                };
                N_VLinearSum(ONE, &eeQ, -ONE, &phiQ_kp1, &tempvQ);
                enorm = IDAQuadWrmsNormUpdate(IDA_mem, enorm, &tempvQ, &ewtQ);
            }

            if IDA_mem.borrow().ida_errconS {
                /* Rename ypS */
                let (tempvS, Ns, eeS, phiS_kp1, ewtS) = {
                    let m = IDA_mem.borrow();
                    (
                        m.ida_ypS.clone(),
                        m.ida_Ns,
                        m.ida_eeS.clone(),
                        m.ida_phiS[(kk + 1) as usize].clone(),
                        m.ida_ewtS.clone(),
                    )
                };

                let _ = N_VLinearSumVectorArray(Ns, ONE, &eeS, -ONE, &phiS_kp1, &tempvS);

                enorm = IDASensWrmsNormUpdate(IDA_mem, enorm, &tempvS, &ewtS, suppressalg);
            }

            if IDA_mem.borrow().ida_errconQS {
                let (Ns, eeQS, phiQS_kp1, tempvQS, ewtQS) = {
                    let m = IDA_mem.borrow();
                    (
                        m.ida_Ns,
                        m.ida_eeQS.clone(),
                        m.ida_phiQS[(kk + 1) as usize].clone(),
                        m.ida_tempvQS.clone(),
                        m.ida_ewtQS.clone(),
                    )
                };

                let _ = N_VLinearSumVectorArray(Ns, ONE, &eeQS, -ONE, &phiQS_kp1, &tempvQS);

                enorm = IDAQuadSensWrmsNormUpdate(IDA_mem, enorm, &tempvQS, &ewtQS);
            }

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
        let (kused, maxord, quadr, sensi, quadr_sensi, Ns) = {
            let m = IDA_mem.borrow();
            (
                m.ida_kused,
                m.ida_maxord,
                m.ida_quadr,
                m.ida_sensi,
                m.ida_quadr_sensi,
                m.ida_Ns,
            )
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

            if quadr {
                let (eeQ, phiQ_next) = {
                    let m = IDA_mem.borrow();
                    (
                        m.ida_eeQ.clone().unwrap(),
                        m.ida_phiQ[(kused + 1) as usize].clone().unwrap(),
                    )
                };
                N_VScale(ONE, &eeQ, &phiQ_next);
            }

            if sensi || quadr_sensi {
                let mut m = IDA_mem.borrow_mut();
                for is in 0..Ns as usize {
                    m.ida_cvals[is] = ONE;
                }
            }

            if sensi {
                let (cvals, eeS, phiS_next) = {
                    let m = IDA_mem.borrow();
                    (
                        m.ida_cvals[0..Ns as usize].to_vec(),
                        m.ida_eeS.clone(),
                        m.ida_phiS[(kused + 1) as usize].clone(),
                    )
                };
                let _ = N_VScaleVectorArray(Ns, &cvals, &eeS, &phiS_next);
            }

            if quadr_sensi {
                let (cvals, eeQS, phiQS_next) = {
                    let m = IDA_mem.borrow();
                    (
                        m.ida_cvals[0..Ns as usize].to_vec(),
                        m.ida_eeQS.clone(),
                        m.ida_phiQS[(kused + 1) as usize].clone(),
                    )
                };
                let _ = N_VScaleVectorArray(Ns, &cvals, &eeQS, &phiQS_next);
            }
        }
    }

    /* Update phi arrays */

    /* To update phi arrays compute X += Z where                  */
    /* X = [ phi[kused], phi[kused-1], phi[kused-2], ... phi[1] ] */
    /* Z = [ ee,         phi[kused],   phi[kused-1], ... phi[0] ] */

    let (quadr, sensi, quadr_sensi, Ns, kused) = {
        let m = IDA_mem.borrow();
        (
            m.ida_quadr,
            m.ida_sensi,
            m.ida_quadr_sensi,
            m.ida_Ns,
            m.ida_kused as usize,
        )
    };

    let (Xvecs, Zvecs) = {
        let mut m = IDA_mem.borrow_mut();

        let mut Zvecs: Vec<N_Vector> = Vec::with_capacity(kused + 1);
        let mut Xvecs: Vec<N_Vector> = Vec::with_capacity(kused + 1);
        Zvecs.push(m.ida_ee.clone().unwrap());
        Xvecs.push(m.ida_phi[kused].clone().unwrap());
        for j in 1..=kused {
            Zvecs.push(m.ida_phi[kused - j + 1].clone().unwrap());
            Xvecs.push(m.ida_phi[kused - j].clone().unwrap());
        }

        /* mirror the C mem state (ida_Xvecs / ida_Zvecs are pure scratch) */
        idac_store_scratch(&mut m.ida_Xvecs, &Xvecs);
        idac_store_scratch(&mut m.ida_Zvecs, &Zvecs);

        (Xvecs, Zvecs)
    };

    /* C passes `ida_Xvecs` for both X and Z; the port passes the same slice so
    the array-pointer equality test selects the identical in-place (axpy)
    branch, preserving the sequential cascade over the phi columns. */
    let _ = N_VLinearSumVectorArray((kused + 1) as i32, ONE, &Xvecs, ONE, &Zvecs, &Xvecs);

    if quadr {
        let (Xvecs, Zvecs) = {
            let mut m = IDA_mem.borrow_mut();

            let mut Zvecs: Vec<N_Vector> = Vec::with_capacity(kused + 1);
            let mut Xvecs: Vec<N_Vector> = Vec::with_capacity(kused + 1);
            Zvecs.push(m.ida_eeQ.clone().unwrap());
            Xvecs.push(m.ida_phiQ[kused].clone().unwrap());
            for j in 1..=kused {
                Zvecs.push(m.ida_phiQ[kused - j + 1].clone().unwrap());
                Xvecs.push(m.ida_phiQ[kused - j].clone().unwrap());
            }

            idac_store_scratch(&mut m.ida_Xvecs, &Xvecs);
            idac_store_scratch(&mut m.ida_Zvecs, &Zvecs);

            (Xvecs, Zvecs)
        };

        let _ = N_VLinearSumVectorArray((kused + 1) as i32, ONE, &Xvecs, ONE, &Zvecs, &Xvecs);
    }

    if sensi {
        let (Xvecs, Zvecs) = {
            let mut m = IDA_mem.borrow_mut();

            let mut Zvecs: Vec<N_Vector> = Vec::new();
            let mut Xvecs: Vec<N_Vector> = Vec::new();
            for is in 0..Ns as usize {
                Zvecs.push(m.ida_eeS[is].clone());
                Xvecs.push(m.ida_phiS[kused][is].clone());
                for j in 1..=kused {
                    Zvecs.push(m.ida_phiS[kused - j + 1][is].clone());
                    Xvecs.push(m.ida_phiS[kused - j][is].clone());
                }
            }

            idac_store_scratch(&mut m.ida_Xvecs, &Xvecs);
            idac_store_scratch(&mut m.ida_Zvecs, &Zvecs);

            (Xvecs, Zvecs)
        };

        let _ = N_VLinearSumVectorArray(Ns * (kused as i32 + 1), ONE, &Xvecs, ONE, &Zvecs, &Xvecs);
    }

    if quadr_sensi {
        let (Xvecs, Zvecs) = {
            let mut m = IDA_mem.borrow_mut();

            let mut Zvecs: Vec<N_Vector> = Vec::new();
            let mut Xvecs: Vec<N_Vector> = Vec::new();
            for is in 0..Ns as usize {
                Zvecs.push(m.ida_eeQS[is].clone());
                Xvecs.push(m.ida_phiQS[kused][is].clone());
                for j in 1..=kused {
                    Zvecs.push(m.ida_phiQS[kused - j + 1][is].clone());
                    Xvecs.push(m.ida_phiQS[kused - j][is].clone());
                }
            }

            idac_store_scratch(&mut m.ida_Xvecs, &Xvecs);
            idac_store_scratch(&mut m.ida_Zvecs, &Zvecs);

            (Xvecs, Zvecs)
        };

        let _ = N_VLinearSumVectorArray(Ns * (kused as i32 + 1), ONE, &Xvecs, ONE, &Zvecs, &Xvecs);
    }
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
 * Norm functions
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
 * IDASensWrmsNorm
 *
 * This routine returns the maximum over the weighted root mean
 * square norm of xS with weight vectors wS:
 *
 *   max { wrms(xS[0],wS[0]) ... wrms(xS[Ns-1],wS[Ns-1]) }
 *
 * Called by IDASensUpdateNorm or directly in the IDA_STAGGERED approach
 * during the NLS solution and before the error test.
 *
 * Declared global for use in the computation of IC for sensitivities.
 */

pub fn IDASensWrmsNorm(
    IDA_mem: &IDAMem,
    xS: &[N_Vector],
    wS: &[N_Vector],
    mask: sunbooleantype,
) -> sunrealtype {
    let Ns = IDA_mem.borrow().ida_Ns;

    /* `ida_cvals` is the C scratch array the vector-array norm writes into;
    move it out of the mem for the duration of the call so no borrow is held
    across a (possibly user-overridden) vector operation. */
    let mut cvals = std::mem::take(&mut IDA_mem.borrow_mut().ida_cvals);

    if mask {
        let id = IDA_mem.borrow().ida_id.clone().unwrap();
        let _ = N_VWrmsNormMaskVectorArray(Ns, xS, wS, &id, &mut cvals);
    } else {
        let _ = N_VWrmsNormVectorArray(Ns, xS, wS, &mut cvals);
    }

    let mut nrm = cvals[0];
    for is in 1..Ns as usize {
        if cvals[is] > nrm {
            nrm = cvals[is];
        }
    }

    IDA_mem.borrow_mut().ida_cvals = cvals;

    nrm
}

/*
 * IDAQuadSensWrmsNorm
 *
 * This routine returns the maximum over the weighted root mean
 * square norm of xQS with weight vectors wQS:
 *
 *   max { wrms(xQS[0],wQS[0]) ... wrms(xQS[Ns-1],wQS[Ns-1]) }
 */

fn IDAQuadSensWrmsNorm(IDA_mem: &IDAMem, xQS: &[N_Vector], wQS: &[N_Vector]) -> sunrealtype {
    let Ns = IDA_mem.borrow().ida_Ns;

    let mut cvals = std::mem::take(&mut IDA_mem.borrow_mut().ida_cvals);

    let _ = N_VWrmsNormVectorArray(Ns, xQS, wQS, &mut cvals);

    let mut nrm = cvals[0];
    for is in 1..Ns as usize {
        if cvals[is] > nrm {
            nrm = cvals[is];
        }
    }

    IDA_mem.borrow_mut().ida_cvals = cvals;

    nrm
}

/*
 * IDAQuadWrmsNormUpdate
 *
 * Updates the norm old_nrm to account for all quadratures.
 */

fn IDAQuadWrmsNormUpdate(
    _IDA_mem: &IDAMem,
    old_nrm: sunrealtype,
    xQ: &N_Vector,
    wQ: &N_Vector,
) -> sunrealtype {
    let qnrm = N_VWrmsNorm(xQ, wQ);
    if old_nrm > qnrm {
        old_nrm
    } else {
        qnrm
    }
}

/*
 * IDASensWrmsNormUpdate
 *
 * Updates the norm old_nrm to account for all sensitivities.
 *
 * This function is declared global since it is used for finding
 * IC for sensitivities,
 */

pub fn IDASensWrmsNormUpdate(
    IDA_mem: &IDAMem,
    old_nrm: sunrealtype,
    xS: &[N_Vector],
    wS: &[N_Vector],
    mask: sunbooleantype,
) -> sunrealtype {
    let snrm = IDASensWrmsNorm(IDA_mem, xS, wS, mask);
    if old_nrm > snrm {
        old_nrm
    } else {
        snrm
    }
}

fn IDAQuadSensWrmsNormUpdate(
    IDA_mem: &IDAMem,
    old_nrm: sunrealtype,
    xQS: &[N_Vector],
    wQS: &[N_Vector],
) -> sunrealtype {
    let qsnrm = IDAQuadSensWrmsNorm(IDA_mem, xQS, wQS);
    if old_nrm > qsnrm {
        old_nrm
    } else {
        qsnrm
    }
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
    let mut glo = std::mem::take(&mut IDA_mem.borrow_mut().ida_glo);
    let retval = idac_call_gfun(IDA_mem, tlo, &phi0, &phi1, &mut glo);
    {
        let mut m = IDA_mem.borrow_mut();
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
    let mut ghi = std::mem::take(&mut IDA_mem.borrow_mut().ida_ghi);
    let retval = idac_call_gfun(IDA_mem, tplus, &yy, &phi1, &mut ghi);
    {
        let mut m = IDA_mem.borrow_mut();
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
    let mut glo = std::mem::take(&mut IDA_mem.borrow_mut().ida_glo);
    let retval = idac_call_gfun(IDA_mem, tlo, &yy, &yp, &mut glo);
    {
        let mut m = IDA_mem.borrow_mut();
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
    let mut ghi = std::mem::take(&mut IDA_mem.borrow_mut().ida_ghi);
    let retval = idac_call_gfun(IDA_mem, tplus, &yy, &yp, &mut ghi);
    {
        let mut m = IDA_mem.borrow_mut();
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
    let mut ghi = std::mem::take(&mut IDA_mem.borrow_mut().ida_ghi);
    let retval = idac_call_gfun(IDA_mem, thi, &yy, &yp, &mut ghi);
    {
        let mut m = IDA_mem.borrow_mut();
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
    last completed iteration, exactly as in C). */
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
    /* The locked move-state-into-locals pattern (ARCHITECTURE "Granular
    borrow rule"), identical in cvode_rs/cvodes_rs/ida_rs: the six
    rootfinding arrays live in locals for the whole Illinois search and are
    written back at the single exit point below. C leaves them readable in
    IDA_mem across `ida_gfun`, so a `g` that re-enters the solver
    (IDAGetRootInfo / IDA{Get,Set}RootDirection) would see empty Vecs here.
    No valid example calls the solver API from inside `g`. */
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
                let retval = idac_call_gfun(IDA_mem, tmid, &yy, &yp, &mut grout);
                IDA_mem.borrow_mut().ida_nge += 1;
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

/*
 * =================================================================
 * Internal DQ approximations for sensitivity RHS
 * =================================================================
 */

/*
 * `IDA_mem->ida_p` is the caller's own parameter array (C stores the
 * POINTER; the port stores a clone of the caller's `SensParams` handle —
 * see `idas_impl::SensParams`), so the perturbations below are visible
 * to the user's `res`/`rhsQ` through their `user_data`, exactly as in C.
 *
 * Both accessors borrow the parameter cell for the duration of one
 * statement only — never across the user callback that follows.
 */

/// C: `psave = IDA_mem->ida_p[which];`
fn ida_p_get(IDA_mem: &IDAMem, which: i32) -> sunrealtype {
    let p = IDA_mem.borrow().ida_p.clone().expect("ida_p set");
    let psave = p.borrow()[which as usize];

    psave
}

/// C: `IDA_mem->ida_p[which] = value;`
fn ida_p_set(IDA_mem: &IDAMem, which: i32, value: sunrealtype) {
    let p = IDA_mem.borrow().ida_p.clone().expect("ida_p set");
    p.borrow_mut()[which as usize] = value;
}

/*
 * IDASensResDQ
 *
 * IDASensRhsDQ computes the residuals of the sensitivity equations
 * by finite differences. It is of type IDASensResFn.
 * Returns 0 if successful, <0 if an unrecoverable failure occurred,
 * >0 for a recoverable error.
 *
 * NOTE: the signature must stay exactly `IDASensResFn` -- `ida_resS` is
 * assigned `Some(IDASensResDQ)` in IDASensInit / IDASensReInit.
 */

pub fn IDASensResDQ(
    Ns: i32,
    t: sunrealtype,
    yy: &N_Vector,
    yp: &N_Vector,
    resval: &N_Vector,
    yyS: &[N_Vector],
    ypS: &[N_Vector],
    resvalS: &[N_Vector],
    user_dataS: &mut Option<Box<dyn std::any::Any>>,
    ytemp: &N_Vector,
    yptemp: &N_Vector,
    restemp: &N_Vector,
) -> i32 {
    let mut retval: i32;

    for is in 0..Ns as usize {
        retval = IDASensRes1DQ(
            Ns,
            t,
            yy,
            yp,
            resval,
            is as i32,
            &yyS[is],
            &ypS[is],
            &resvalS[is],
            user_dataS,
            ytemp,
            yptemp,
            restemp,
        );
        if retval != 0 {
            return retval;
        }
    }
    0
}

/*
 * IDASensRes1DQ
 *
 * IDASensRes1DQ computes the residual of the is-th sensitivity
 * equation by finite differences.
 *
 * Returns 0 if successful or the return value of res if res fails
 * (<0 if res fails unrecoverably, >0 if res has a recoverable error).
 *
 * `Ns` is `SUNDIALS_MAYBE_UNUSED` in C, hence `_Ns` here.
 */

fn IDASensRes1DQ(
    _Ns: i32,
    t: sunrealtype,
    yy: &N_Vector,
    yp: &N_Vector,
    resval: &N_Vector,
    is: i32,
    yyS: &N_Vector,
    ypS: &N_Vector,
    resvalS: &N_Vector,
    user_dataS: &mut Option<Box<dyn std::any::Any>>,
    ytemp: &N_Vector,
    yptemp: &N_Vector,
    restemp: &N_Vector,
) -> i32 {
    let method: i32;
    let mut retval: i32;
    let psave: sunrealtype;
    let pbari: sunrealtype;
    let del: sunrealtype;
    let rdel: sunrealtype;
    let Delp: sunrealtype;
    let rDelp: sunrealtype;
    let Dely: sunrealtype;
    let rDely: sunrealtype;
    let ratio: sunrealtype;

    /* user_dataS points to IDA_mem (a boxed IDAMem handle clone; C's cast of
    a NULL/foreign pointer is UB -> deterministic panic) */
    let IDA_mem = user_dataS
        .as_mut()
        .and_then(|b| b.downcast_ref::<IDAMem>())
        .cloned()
        .expect("IDASensRes1DQ user_dataS holds IDAMem");
    let IDA_mem = &IDA_mem;

    /* Set base perturbation del */
    {
        let m = IDA_mem.borrow();
        del = SUNRsqrt(SUNMAX(m.ida_rtol, m.ida_uround));
    }
    rdel = ONE / del;

    pbari = IDA_mem.borrow().ida_pbar[is as usize];

    let which = IDA_mem.borrow().ida_plist[is as usize];

    psave = ida_p_get(IDA_mem, which);

    Delp = pbari * del;
    rDelp = ONE / Delp;
    let ewt = IDA_mem.borrow().ida_ewt.clone().unwrap();
    let norms = N_VWrmsNorm(yyS, &ewt) * pbari;
    rDely = SUNMAX(norms, rdel) / pbari;
    Dely = ONE / rDely;

    let (DQrhomax, DQtype) = {
        let m = IDA_mem.borrow();
        (m.ida_DQrhomax, m.ida_DQtype)
    };

    if DQrhomax == ZERO {
        /* No switching */
        method = if DQtype == IDA_CENTERED {
            CENTERED1
        } else {
            FORWARD1
        };
    } else {
        /* switch between simultaneous/separate DQ */
        ratio = Dely * rDelp;
        if SUNMAX(ONE / ratio, ratio) <= DQrhomax {
            method = if DQtype == IDA_CENTERED {
                CENTERED1
            } else {
                FORWARD1
            };
        } else {
            method = if DQtype == IDA_CENTERED {
                CENTERED2
            } else {
                FORWARD2
            };
        }
    }

    match method {
        CENTERED1 => {
            let Del = SUNMIN(Dely, Delp);
            let r2Del = HALF / Del;

            /* Forward perturb y, y' and parameter */
            N_VLinearSum(Del, yyS, ONE, yy, ytemp);
            N_VLinearSum(Del, ypS, ONE, yp, yptemp);
            ida_p_set(IDA_mem, which, psave + Del);

            /* Save residual in resvalS */
            retval = idac_call_res(IDA_mem, t, ytemp, yptemp, resvalS);
            IDA_mem.borrow_mut().ida_nreS += 1;
            if retval != 0 {
                return retval;
            }

            /* Backward perturb y, y' and parameter */
            N_VLinearSum(-Del, yyS, ONE, yy, ytemp);
            N_VLinearSum(-Del, ypS, ONE, yp, yptemp);
            ida_p_set(IDA_mem, which, psave - Del);

            /* Save residual in restemp */
            retval = idac_call_res(IDA_mem, t, ytemp, yptemp, restemp);
            IDA_mem.borrow_mut().ida_nreS += 1;
            if retval != 0 {
                return retval;
            }

            /* Estimate the residual for the i-th sensitivity equation */
            N_VLinearSum(r2Del, resvalS, -r2Del, restemp, resvalS);
        }

        CENTERED2 => {
            let r2Delp = HALF / Delp;
            let r2Dely = HALF / Dely;

            /* Forward perturb y and y' */
            N_VLinearSum(Dely, yyS, ONE, yy, ytemp);
            N_VLinearSum(Dely, ypS, ONE, yp, yptemp);

            /* Save residual in resvalS */
            retval = idac_call_res(IDA_mem, t, ytemp, yptemp, resvalS);
            IDA_mem.borrow_mut().ida_nreS += 1;
            if retval != 0 {
                return retval;
            }

            /* Backward perturb y and y' */
            N_VLinearSum(-Dely, yyS, ONE, yy, ytemp);
            N_VLinearSum(-Dely, ypS, ONE, yp, yptemp);

            /* Save residual in restemp */
            retval = idac_call_res(IDA_mem, t, ytemp, yptemp, restemp);
            IDA_mem.borrow_mut().ida_nreS += 1;
            if retval != 0 {
                return retval;
            }

            /* Save the first difference quotient in resvalS */
            N_VLinearSum(r2Dely, resvalS, -r2Dely, restemp, resvalS);

            /* Forward perturb parameter */
            ida_p_set(IDA_mem, which, psave + Delp);

            /* Save residual in ytemp */
            retval = idac_call_res(IDA_mem, t, yy, yp, ytemp);
            IDA_mem.borrow_mut().ida_nreS += 1;
            if retval != 0 {
                return retval;
            }

            /* Backward perturb parameter */
            ida_p_set(IDA_mem, which, psave - Delp);

            /* Save residual in yptemp */
            retval = idac_call_res(IDA_mem, t, yy, yp, yptemp);
            IDA_mem.borrow_mut().ida_nreS += 1;
            if retval != 0 {
                return retval;
            }

            /* Save the second difference quotient in restemp */
            N_VLinearSum(r2Delp, ytemp, -r2Delp, yptemp, restemp);

            /* Add the difference quotients for the sensitivity residual */
            N_VLinearSum(ONE, resvalS, ONE, restemp, resvalS);
        }

        FORWARD1 => {
            let Del = SUNMIN(Dely, Delp);
            let rDel = ONE / Del;

            /* Forward perturb y, y' and parameter */
            N_VLinearSum(Del, yyS, ONE, yy, ytemp);
            N_VLinearSum(Del, ypS, ONE, yp, yptemp);
            ida_p_set(IDA_mem, which, psave + Del);

            /* Save residual in resvalS */
            retval = idac_call_res(IDA_mem, t, ytemp, yptemp, resvalS);
            IDA_mem.borrow_mut().ida_nreS += 1;
            if retval != 0 {
                return retval;
            }

            /* Estimate the residual for the i-th sensitivity equation */
            N_VLinearSum(rDel, resvalS, -rDel, resval, resvalS);
        }

        FORWARD2 => {
            /* Forward perturb y and y' */
            N_VLinearSum(Dely, yyS, ONE, yy, ytemp);
            N_VLinearSum(Dely, ypS, ONE, yp, yptemp);

            /* Save residual in resvalS */
            retval = idac_call_res(IDA_mem, t, ytemp, yptemp, resvalS);
            IDA_mem.borrow_mut().ida_nreS += 1;
            if retval != 0 {
                return retval;
            }

            /* Save the first difference quotient in resvalS */
            N_VLinearSum(rDely, resvalS, -rDely, resval, resvalS);

            /* Forward perturb parameter */
            ida_p_set(IDA_mem, which, psave + Delp);

            /* Save residual in restemp */
            retval = idac_call_res(IDA_mem, t, yy, yp, restemp);
            IDA_mem.borrow_mut().ida_nreS += 1;
            if retval != 0 {
                return retval;
            }

            /* Save the second difference quotient in restemp */
            N_VLinearSum(rDelp, restemp, -rDelp, resval, restemp);

            /* Add the difference quotients for the sensitivity residual */
            N_VLinearSum(ONE, resvalS, ONE, restemp, resvalS);
        }

        _ => {}
    }

    /* Restore original value of parameter */
    ida_p_set(IDA_mem, which, psave);

    0
}

/* IDAQuadSensRhsInternalDQ   - internal IDAQuadSensRhsFn
 *
 * IDAQuadSensRhsInternalDQ computes right hand side of all quadrature
 * sensitivity equations by finite differences. All work is actually
 * done in IDAQuadSensRhs1InternalDQ.
 *
 * NOTE: the signature must stay exactly `IDAQuadSensRhsFn` --
 * `ida_rhsQS` is assigned `Some(IDAQuadSensRhsInternalDQ)` in
 * IDACreate / IDAQuadSensInit.
 */

fn IDAQuadSensRhsInternalDQ(
    Ns: i32,
    t: sunrealtype,
    yy: &N_Vector,
    yp: &N_Vector,
    yyS: &[N_Vector],
    ypS: &[N_Vector],
    rrQ: &N_Vector,
    resvalQS: &[N_Vector],
    ida_mem: &mut Option<Box<dyn std::any::Any>>,
    yytmp: &N_Vector,
    yptmp: &N_Vector,
    tmpQS: &N_Vector,
) -> i32 {
    let mut retval: i32;

    /* ida_mem is passed here as user data (a boxed IDAMem handle clone) */
    let IDA_mem = ida_mem
        .as_mut()
        .and_then(|b| b.downcast_ref::<IDAMem>())
        .cloned()
        .expect("IDAQuadSensRhsInternalDQ ida_mem holds IDAMem");
    let IDA_mem = &IDA_mem;

    for is in 0..Ns as usize {
        retval = IDAQuadSensRhs1InternalDQ(
            IDA_mem,
            is as i32,
            t,
            yy,
            yp,
            &yyS[is],
            &ypS[is],
            rrQ,
            &resvalQS[is],
            yytmp,
            yptmp,
            tmpQS,
        );
        if retval != 0 {
            return retval;
        }
    }

    0
}

fn IDAQuadSensRhs1InternalDQ(
    IDA_mem: &IDAMem,
    is: i32,
    t: sunrealtype,
    yy: &N_Vector,
    yp: &N_Vector,
    yyS: &N_Vector,
    ypS: &N_Vector,
    resvalQ: &N_Vector,
    resvalQS: &N_Vector,
    yytmp: &N_Vector,
    yptmp: &N_Vector,
    tmpQS: &N_Vector,
) -> i32 {
    let mut retval: i32;
    let method: i32;
    let mut nfel: i32 = 0;
    let psave: sunrealtype;
    let pbari: sunrealtype;
    let del: sunrealtype;
    /* C reuses `rdel` in the FORWARD1 branch (`rdel = ONE / Del`), so it must
    stay mutable here. */
    let mut rdel: sunrealtype;
    let Delp: sunrealtype;
    let Dely: sunrealtype;
    let rDely: sunrealtype;

    {
        let m = IDA_mem.borrow();
        del = SUNRsqrt(SUNMAX(m.ida_rtol, m.ida_uround));
    }
    rdel = ONE / del;

    pbari = IDA_mem.borrow().ida_pbar[is as usize];

    let which = IDA_mem.borrow().ida_plist[is as usize];

    psave = ida_p_get(IDA_mem, which);

    Delp = pbari * del;
    let ewt = IDA_mem.borrow().ida_ewt.clone().unwrap();
    let norms = N_VWrmsNorm(yyS, &ewt) * pbari;
    rDely = SUNMAX(norms, rdel) / pbari;
    Dely = ONE / rDely;

    let DQtype = IDA_mem.borrow().ida_DQtype;
    method = if DQtype == IDA_CENTERED {
        CENTERED1
    } else {
        FORWARD1
    };

    match method {
        CENTERED1 => {
            let Del = SUNMIN(Dely, Delp);
            let r2Del = HALF / Del;

            N_VLinearSum(ONE, yy, Del, yyS, yytmp);
            N_VLinearSum(ONE, yp, Del, ypS, yptmp);
            ida_p_set(IDA_mem, which, psave + Del);

            retval = idac_call_rhsQ(IDA_mem, t, yytmp, yptmp, resvalQS);
            nfel += 1;
            if retval != 0 {
                return retval;
            }

            N_VLinearSum(-Del, yyS, ONE, yy, yytmp);
            N_VLinearSum(-Del, ypS, ONE, yp, yptmp);

            ida_p_set(IDA_mem, which, psave - Del);

            retval = idac_call_rhsQ(IDA_mem, t, yytmp, yptmp, tmpQS);
            nfel += 1;
            if retval != 0 {
                return retval;
            }

            N_VLinearSum(r2Del, resvalQS, -r2Del, tmpQS, resvalQS);
        }

        FORWARD1 => {
            let Del = SUNMIN(Dely, Delp);
            rdel = ONE / Del;

            N_VLinearSum(ONE, yy, Del, yyS, yytmp);
            N_VLinearSum(ONE, yp, Del, ypS, yptmp);
            ida_p_set(IDA_mem, which, psave + Del);

            retval = idac_call_rhsQ(IDA_mem, t, yytmp, yptmp, resvalQS);
            nfel += 1;
            if retval != 0 {
                return retval;
            }

            N_VLinearSum(rdel, resvalQS, -rdel, resvalQ, resvalQS);
        }

        _ => {}
    }

    ida_p_set(IDA_mem, which, psave);
    /* Increment counter nrQeS */
    IDA_mem.borrow_mut().ida_nrQeS += nfel as i64;

    0
}

/*
 * =================================================================
 * Regression tests
 * =================================================================
 */

#[cfg(test)]
mod tests {
    use super::*;
    use sundials_core::sundials_libm::SunMath;

    use crate::idas_io::{IDASetSensParams, IDASetUserData};
    use crate::idas_ls::{IDASetLinearSolver, IDALS_SUCCESS};
    use sundials_core::nvector_serial::N_VNew_Serial;
    use sundials_core::sundials_context::SUNContext_Create;
    use sundials_core::sunlinsol_dense::SUNLinSol_Dense;
    use sundials_core::sunmatrix_dense::SUNDenseMatrix;

    /* -----------------------------------------------------------------
     * Internal-DQ forward sensitivity relies on ALIASING: C stores the
     * caller's `p` POINTER in `IDA_mem->ida_p` and `IDASensRes1DQ`
     * perturbs `ida_p[which]` in place around each call to the user's
     * `res`, which reads the very same array through its `user_data`. The
     * port shares the array as a `SensParams` handle (ARCHITECTURE §8);
     * this test is the executable proof that the perturbation reaches the
     * callback.
     *
     * Problem:  F(t,y,y',p) = y' + p*y = 0,  y(0) = 1,  p = 2
     *   exact:  y(t)     = exp(-p t)
     *           dy/dp(t) = -t exp(-p t)
     * With a private copy of `p` the callback would never see the
     * perturbation, so dF/dp would be identically zero and the computed
     * sensitivity would stay exactly 0 for all time.
     * -----------------------------------------------------------------*/

    const P0: sunrealtype = 2.0;
    const TEND: sunrealtype = 1.0;

    struct SensTestData {
        /* the shared parameter array — the solver holds a clone of this
        very handle (C: the same `sunrealtype*`) */
        p: SensParams,
        /* extreme parameter values observed by the residual callback */
        pmin: sunrealtype,
        pmax: sunrealtype,
    }

    fn sens_test_res(
        _tres: sunrealtype,
        yy: &N_Vector,
        yp: &N_Vector,
        rr: &N_Vector,
        user_data: &mut Option<Box<dyn Any>>,
    ) -> i32 {
        let data = user_data
            .as_mut()
            .and_then(|b| b.downcast_mut::<SensTestData>())
            .expect("user_data is SensTestData");

        /* read the parameter exactly as a C callback would read
        data->p[0]; the borrow ends with this statement */
        let p = data.p.borrow()[0];

        if p < data.pmin {
            data.pmin = p;
        }
        if p > data.pmax {
            data.pmax = p;
        }

        let yval = N_VGetArrayPointer(yy).expect("N_VGetArrayPointer")[0];
        let ypval = N_VGetArrayPointer(yp).expect("N_VGetArrayPointer")[0];
        N_VGetArrayPointer(rr).expect("N_VGetArrayPointer")[0] = ypval + p * yval;

        0
    }

    #[test]
    fn internal_dq_sensitivity_sees_perturbed_parameters() {
        let mut sunctx: Option<SUNContext> = None;
        assert_eq!(SUNContext_Create(SUN_COMM_NULL, &mut sunctx), 0);
        let sunctx = sunctx.expect("SUNContext_Create");

        /* the parameter array the user owns and the solver shares */
        let p: SensParams = Rc::new(RefCell::new(vec![P0]));

        let yy = N_VNew_Serial(1, &sunctx).expect("N_VNew_Serial");
        let yp = N_VNew_Serial(1, &sunctx).expect("N_VNew_Serial");
        N_VGetArrayPointer(&yy).expect("N_VGetArrayPointer")[0] = ONE;
        N_VGetArrayPointer(&yp).expect("N_VGetArrayPointer")[0] = -P0;

        let ida_mem = IDACreate(&sunctx).expect("IDACreate");

        assert_eq!(
            IDAInit(&ida_mem, sens_test_res, ZERO, &yy, &yp),
            IDA_SUCCESS
        );
        assert_eq!(IDASStolerances(&ida_mem, 1.0e-8, 1.0e-10), IDA_SUCCESS);

        /* the user data holds a CLONE of the same handle */
        let data = SensTestData {
            p: p.clone(),
            pmin: P0,
            pmax: P0,
        };
        assert_eq!(IDASetUserData(&ida_mem, Some(Box::new(data))), IDA_SUCCESS);

        let A = SUNDenseMatrix(1, 1, &sunctx).expect("SUNDenseMatrix");
        let LS = SUNLinSol_Dense(&yy, &A, &sunctx).expect("SUNLinSol_Dense");
        assert_eq!(IDASetLinearSolver(&ida_mem, &LS, Some(&A)), IDALS_SUCCESS);

        /* one sensitivity, computed by the INTERNAL DQ routine (fS = None);
        dy/dp(0) = 0 and dy'/dp(0) = -y(0) = -1 */
        let yS = N_VCloneVectorArray(1, &yy).expect("N_VCloneVectorArray");
        let ypS = N_VCloneVectorArray(1, &yy).expect("N_VCloneVectorArray");
        N_VConst(ZERO, &yS[0]);
        N_VGetArrayPointer(&ypS[0]).expect("N_VGetArrayPointer")[0] = -ONE;
        assert_eq!(
            IDASensInit(&ida_mem, 1, IDA_SIMULTANEOUS, None, &yS, &ypS),
            IDA_SUCCESS
        );
        assert_eq!(IDASensEEtolerances(&ida_mem), IDA_SUCCESS);

        /* C: IDASetSensParams(ida_mem, data->params, pbar, NULL) */
        let pbar = [P0];
        assert_eq!(
            IDASetSensParams(&ida_mem, Some(p.clone()), Some(&pbar[..]), None),
            IDA_SUCCESS
        );

        let mut t = ZERO;
        let flag = IDASolve(&ida_mem, TEND, &mut t, &yy, &yp, IDA_NORMAL);
        assert_eq!(flag, IDA_SUCCESS, "IDASolve failed with flag {flag}");
        assert_eq!(t, TEND);

        /* state: y(TEND) = exp(-p*TEND) */
        let yend = N_VGetArrayPointer(&yy).expect("N_VGetArrayPointer")[0];
        let y_exact = (-P0 * TEND).sun_exp();
        assert!(
            SUNRabs(yend - y_exact) <= 1.0e-6 * y_exact,
            "state wrong: got {yend}, expected {y_exact}"
        );

        /* sensitivity: dy/dp(TEND) = -TEND*exp(-p*TEND) */
        let mut tS = ZERO;
        assert_eq!(IDAGetSens(&ida_mem, &mut tS, &yS), IDA_SUCCESS);
        let s = N_VGetArrayPointer(&yS[0]).expect("N_VGetArrayPointer")[0];
        let s_exact = -TEND * (-P0 * TEND).sun_exp();

        /* the defect signature: with an unshared copy of `p` this is 0 */
        assert!(
            s != ZERO,
            "sensitivity is identically zero — the DQ perturbation never reached the residual callback"
        );
        assert!(
            SUNRabs(s - s_exact) <= 1.0e-4 * SUNRabs(s_exact),
            "sensitivity wrong: got {s}, expected {s_exact}"
        );

        /* direct proof of the aliasing: the callback saw p perturbed both
        ways (IDA_CENTERED is the default DQtype) */
        let mut user_data: Option<Box<dyn Any>> = None;
        std::mem::swap(&mut ida_mem.borrow_mut().ida_user_data, &mut user_data);
        let data = user_data
            .as_ref()
            .and_then(|b| b.downcast_ref::<SensTestData>())
            .expect("user_data is SensTestData");
        assert!(
            data.pmax > P0 && data.pmin < P0,
            "residual never observed a perturbed parameter (saw [{}, {}], p = {P0})",
            data.pmin,
            data.pmax
        );

        /* and the array the caller still owns was restored exactly */
        assert_eq!(p.borrow()[0], P0);
    }
}
