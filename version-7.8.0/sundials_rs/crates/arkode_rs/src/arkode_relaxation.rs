//! Port of `src/arkode/arkode_relaxation.c` (ARKODE's relaxation-in-time
//! functionality). `ARKodeRelaxMemRec`, `ARKRelaxDeltaEFn`,
//! `ARKRelaxGetOrderFn`, the `ARK_RELAX_DEFAULT_*` / `ARK_RELAX_*_RECV`
//! constants and `MSG_RELAX_MEM_NULL` (all from
//! `arkode_relaxation_impl.h`) live in the frozen contract
//! (`arkode_impl.rs`), because `arkode_impl.h` `#include`s that header and
//! `ARKodeMemRec` embeds the record.
//!
//! Temporary vectors utilized in the functions below:
//!   tempv2 - holds delta_y, the update direction vector
//!   tempv3 - holds y_relax, the relaxed solution vector
//!   tempv4 - holds J_relax, the Jacobian of the relaxation function
//!
//! Binding notes:
//! * `void* arkode_mem` -> `&ARKodeMem`; every `arkode_mem == NULL` guard is
//!   unrepresentable and drops out (`arkRelaxAccessMem` survives as the
//!   `relax_mem == NULL` presence check, message and flag unchanged).
//! * C `arkRelaxSolve(ark_mem, relax_mem, relax_val_out)` drops its
//!   `relax_mem` parameter: the record lives inside `ark_mem` and must stay
//!   there, because the stepper-supplied `delta_e_fn` and `get_order_fn` are
//!   handed `ark_mem` and re-enter it. Every C `relax_mem->…` access becomes
//!   a granular scoped borrow through [`relax_get`] / [`relax_set`] at exactly
//!   the C read/write point, so field-read order (and the arithmetic) is
//!   unchanged. Same treatment as `arkode_splittingstep.rs`'s `step_mem`.
//! * Where C hands a callee the address of a `relax_mem` field
//!   (`&relax_mem->res`, `&relax_mem->jac`, `&relax_mem->num_relax_jac_evals`,
//!   `&relax_mem->delta_e`), the port copies the field out, passes `&mut` on
//!   the local, and writes the result back into the record IMMEDIATELY after
//!   the call -- before any flag test -- so the failure paths observe exactly
//!   the values C leaves behind.
//! * `user_data` is `Option::take`n around every `relax_fn` / `relax_jac_fn`
//!   invocation and restored on every path.
//! * `SUNRpowerI` (never `f64::powi`) for the C `SUNRpowerI` in `arkRelax`.
//! * `SUNLogDebug` / `SUNLogExtraDebug` / `SUNLogExtraDebugVec` compile away at
//!   `SUNDIALS_LOGGING_LEVEL=2` and are omitted.

use crate::arkode_impl::*;
use sundials_core::sundials_math::{SUNRabs, SUNRcopysign, SUNRpowerI, SUNRsamesign, SUNMIN};
use sundials_core::sundials_nvector::{N_VDotProd, N_VLinearSum};
use sundials_core::sundials_types::*;
use sundials_core::sundials_utils::{sunfprintf_long, SUNFile};

/* =============================================================================
 * Private Functions
 * ===========================================================================*/

/// Read one or more fields of `ark_mem.relax_mem` under a short-lived borrow
/// (C `relax_mem->field`). Panics if the relaxation memory is unset, which C
/// would reach only by dereferencing NULL (deviation class 5); the reachable
/// NULL cases are guarded by `arkRelaxAccessMem` / `arkRelax` exactly as in C.
fn relax_get<T>(ark_mem: &ARKodeMem, f: impl FnOnce(&ARKodeRelaxMemRec) -> T) -> T {
    let m = ark_mem.borrow();
    f(m.relax_mem.as_ref().expect("relax_mem"))
}

/// Write one or more fields of `ark_mem.relax_mem` under a short-lived borrow.
fn relax_set(ark_mem: &ARKodeMem, f: impl FnOnce(&mut ARKodeRelaxMemRec)) {
    let mut m = ark_mem.borrow_mut();
    f(m.relax_mem.as_mut().expect("relax_mem"));
}

/* Access the ARKODE and relaxation memory structures */
fn arkRelaxAccessMem(ark_mem: &ARKodeMem, fname: &str) -> i32 {
    /* the C `arkode_mem == NULL` branch is unrepresentable */
    if ark_mem.borrow().relax_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_RELAX_MEM_NULL,
            line!() as i32,
            fname,
            file!(),
            MSG_RELAX_MEM_NULL,
        );
        return ARK_RELAX_MEM_NULL;
    }

    ARK_SUCCESS
}

/* Evaluates the relaxation residual function */
fn arkRelaxResidual(
    relax_param: sunrealtype,
    relax_res: &mut sunrealtype,
    ark_mem: &ARKodeMem,
) -> i32 {
    let (e_old, delta_e, relax_fn) = relax_get(ark_mem, |r| {
        (r.e_old, r.delta_e, r.relax_fn.expect("relax_fn"))
    });
    let (yn, delta_y, y_relax) = {
        let m = ark_mem.borrow();
        (
            m.yn.clone().expect("yn"),
            m.tempv2.clone().expect("tempv2"),
            m.tempv3.clone().expect("tempv3"),
        )
    };

    /* y_relax = y_n + r * delta_y */
    N_VLinearSum(ONE, &yn, relax_param, &delta_y, &y_relax);

    /* Evaluate entropy function */
    let mut user_data = ark_mem.borrow_mut().user_data.take();
    let retval = relax_fn(&y_relax, relax_res, &mut user_data);
    ark_mem.borrow_mut().user_data = user_data;
    relax_set(ark_mem, |r| r.num_relax_fn_evals += 1);
    if retval < 0 {
        return ARK_RELAX_FUNC_FAIL;
    }
    if retval > 0 {
        return ARK_RELAX_FUNC_RECV;
    }

    /* Compute relaxation residual */
    *relax_res = *relax_res - e_old - relax_param * delta_e;

    ARK_SUCCESS
}

/* Evaluates the Jacobian of the relaxation residual function */
fn arkRelaxResidualJacobian(
    relax_param: sunrealtype,
    relax_jac: &mut sunrealtype,
    ark_mem: &ARKodeMem,
) -> i32 {
    let (delta_e, relax_jac_fn) = relax_get(ark_mem, |r| {
        (r.delta_e, r.relax_jac_fn.expect("relax_jac_fn"))
    });
    let (yn, delta_y, y_relax, J_relax) = {
        let m = ark_mem.borrow();
        (
            m.yn.clone().expect("yn"),
            m.tempv2.clone().expect("tempv2"),
            m.tempv3.clone().expect("tempv3"),
            m.tempv4.clone().expect("tempv4"),
        )
    };

    /* y_relax = y_n + r * delta_y */
    N_VLinearSum(ONE, &yn, relax_param, &delta_y, &y_relax);

    /* Evaluate Jacobian of entropy functions */
    let mut user_data = ark_mem.borrow_mut().user_data.take();
    let retval = relax_jac_fn(&y_relax, &J_relax, &mut user_data);
    ark_mem.borrow_mut().user_data = user_data;
    relax_set(ark_mem, |r| r.num_relax_jac_evals += 1);
    if retval < 0 {
        return ARK_RELAX_JAC_FAIL;
    }
    if retval > 0 {
        return ARK_RELAX_JAC_RECV;
    }

    /* Compute relaxation residual Jacobian */
    *relax_jac = N_VDotProd(&delta_y, &J_relax);
    *relax_jac -= delta_e;

    ARK_SUCCESS
}

/* Solve the relaxation residual equation using Newton's method */
fn arkRelaxNewtonSolve(ark_mem: &ARKodeMem) -> i32 {
    let mut i: i32 = 0;
    while i < relax_get(ark_mem, |r| r.max_iters) {
        /* Compute the current residual */
        let (relax_param, mut res) = relax_get(ark_mem, |r| (r.relax_param, r.res));
        let retval = arkRelaxResidual(relax_param, &mut res, ark_mem);
        relax_set(ark_mem, |r| r.res = res);
        if retval != 0 {
            return retval;
        }

        /* Check for convergence */
        let (res, res_tol) = relax_get(ark_mem, |r| (r.res, r.res_tol));
        if SUNRabs(res) < res_tol {
            return ARK_SUCCESS;
        }

        /* Compute Jacobian */
        let (relax_param, mut jac) = relax_get(ark_mem, |r| (r.relax_param, r.jac));
        let retval = arkRelaxResidualJacobian(relax_param, &mut jac, ark_mem);
        relax_set(ark_mem, |r| r.jac = jac);
        if retval != 0 {
            return retval;
        }

        /* Update step length tolerance and solution */
        let (rel_tol, relax_param, abs_tol, res, jac) = relax_get(ark_mem, |r| {
            (r.rel_tol, r.relax_param, r.abs_tol, r.res, r.jac)
        });
        let tol = rel_tol * SUNRabs(relax_param) + abs_tol;

        let delta = res / jac;
        relax_set(ark_mem, |r| r.relax_param -= delta);

        /* Update cumulative iteration count */
        relax_set(ark_mem, |r| r.nls_iters += 1);

        /* Check for small update */
        if SUNRabs(delta) < tol {
            return ARK_SUCCESS;
        }

        i += 1;
    }

    ARK_RELAX_SOLVE_RECV
}

/* Solve the relaxation residual equation using Brent's method */
fn arkRelaxBrentSolve(ark_mem: &ARKodeMem) -> i32 {
    /* previous solution and function value */
    let mut xa: sunrealtype;
    let mut fa: sunrealtype = ZERO;
    /* current solution and function value */
    let mut xb: sunrealtype;
    let mut fb: sunrealtype = ZERO;
    /* together brac and curr bracket zero */
    let mut xc: sunrealtype;
    let mut fc: sunrealtype;
    /* midpoint between brac and curr */
    let mut xm: sunrealtype;
    /* previous iteration update */
    let mut old_update: sunrealtype;
    /* new iteration update */
    let mut new_update: sunrealtype;
    /* iteration tolerance */
    let mut tol: sunrealtype;
    /* temporary values */
    let mut pt: sunrealtype;
    let mut qt: sunrealtype;
    let mut rt: sunrealtype;
    let mut st: sunrealtype;

    /* Compute interval that brackets the root */
    let relax_param = relax_get(ark_mem, |r| r.relax_param);
    xa = 0.9 * relax_param;
    xb = 1.1 * relax_param;

    for _i in 0..10 {
        /* Compute relaxation residual */
        let retval = arkRelaxResidual(xa, &mut fa, ark_mem);
        relax_set(ark_mem, |r| r.num_relax_fn_evals += 1);
        if retval < 0 {
            return ARK_RELAX_FUNC_FAIL;
        }
        if retval > 0 {
            return ARK_RELAX_FUNC_RECV;
        }

        /* Check if we got lucky */
        if SUNRabs(fa) < relax_get(ark_mem, |r| r.res_tol) {
            relax_set(ark_mem, |r| {
                r.res = fa;
                r.relax_param = xa;
            });
            return ARK_SUCCESS;
        }

        if fa < ZERO {
            break;
        }

        fb = fa;
        xb = xa;
        xa *= 0.9;
    }
    if fa > ZERO {
        return ARK_RELAX_SOLVE_RECV;
    }

    for _i in 0..10 {
        /* Compute relaxation residual */
        let retval = arkRelaxResidual(xb, &mut fb, ark_mem);
        relax_set(ark_mem, |r| r.num_relax_fn_evals += 1);
        if retval < 0 {
            return ARK_RELAX_FUNC_FAIL;
        }
        if retval > 0 {
            return ARK_RELAX_FUNC_RECV;
        }

        /* Check if we got lucky */
        if SUNRabs(fb) < relax_get(ark_mem, |r| r.res_tol) {
            relax_set(ark_mem, |r| {
                r.res = fb;
                r.relax_param = xb;
            });
            return ARK_SUCCESS;
        }

        if fb > ZERO {
            break;
        }

        fa = fb;
        xa = xb;
        xb *= 1.1;
    }
    if fb < ZERO {
        return ARK_RELAX_SOLVE_RECV;
    }

    /* Initialize values bracketing values to lower bound and updates */
    xc = xa;
    fc = fa;

    old_update = ZERO;
    new_update = ZERO;

    /* Find root */
    let mut i: i32 = 0;
    while i < relax_get(ark_mem, |r| r.max_iters) {
        /* Ensure xc and xb bracket zero */
        if SUNRsamesign(fc, fb) {
            xc = xa;
            fc = fa;
            new_update = xb - xa;
            old_update = new_update;
        }

        /* Ensure xb is closer to zero than xc */
        if SUNRabs(fb) > SUNRabs(fc) {
            xa = xb;
            xb = xc;
            xc = xa;

            fa = fb;
            fb = fc;
            fc = fa;
        }

        /* Update tolerance */
        let (rel_tol, abs_tol) = relax_get(ark_mem, |r| (r.rel_tol, r.abs_tol));
        tol = rel_tol * SUNRabs(xb) + HALF * abs_tol;

        /* Compute midpoint for bisection */
        xm = HALF * (xc - xb);

        /* Check for convergence */
        if SUNRabs(xm) < tol || SUNRabs(fb) < relax_get(ark_mem, |r| r.res_tol) {
            relax_set(ark_mem, |r| {
                r.res = fb;
                r.relax_param = xb;
            });
            return ARK_SUCCESS;
        }

        /* Compute iteration update */
        if SUNRabs(old_update) >= tol && SUNRabs(fb) < SUNRabs(fa) {
            /* Converging sufficiently fast, interpolate solution */
            st = fb / fa;

            if xa == xc {
                /* Two unique values available, try linear interpolant (secant) */
                pt = TWO * xm * st;
                qt = ONE - st;
            } else {
                /* Three unique values available, try inverse quadratic interpolant */
                qt = fa / fc;
                rt = fb / fc;
                pt = st * (TWO * xm * qt * (qt - rt) - (xb - xa) * (rt - ONE));
                qt = (qt - ONE) * (rt - ONE) * (st - ONE);
            }

            /* Ensure updates produce values within [xc, xb] or [xb, xc] */
            if pt > ZERO {
                qt = -qt;
            } else {
                pt = -pt;
            }

            /* Check if interpolant is acceptable, otherwise use bisection */
            st = THREE * xm * qt - SUNRabs(tol * qt);
            rt = SUNRabs(old_update * qt);

            if TWO * pt < SUNMIN(st, rt) {
                old_update = new_update;
                new_update = pt / qt;
            } else {
                new_update = xm;
                old_update = xm;
            }
        } else {
            /* Converging too slowly, use bisection */
            new_update = xm;
            old_update = xm;
        }

        /* Update solution */
        xa = xb;
        fa = fb;

        /* If update is small, use tolerance in bisection direction */
        if SUNRabs(new_update) > tol {
            xb += new_update;
        } else {
            xb += SUNRcopysign(tol, xm);
        }

        /* Compute relaxation residual */
        let retval = arkRelaxResidual(xb, &mut fb, ark_mem);
        relax_set(ark_mem, |r| r.num_relax_fn_evals += 1);
        if retval < 0 {
            return ARK_RELAX_FUNC_FAIL;
        }
        if retval > 0 {
            return ARK_RELAX_FUNC_RECV;
        }

        i += 1;
    }

    ARK_RELAX_SOLVE_RECV
}

/* Compute and apply relaxation parameter */
fn arkRelaxSolve(ark_mem: &ARKodeMem, relax_val_out: &mut sunrealtype) -> i32 {
    /* Get the change in entropy (uses temp vectors 2 and 3) */
    let (delta_e_fn, relax_jac_fn) = relax_get(ark_mem, |r| {
        (r.delta_e_fn.expect("delta_e_fn"), r.relax_jac_fn)
    });
    let (mut evals_out, mut delta_e) = relax_get(ark_mem, |r| (r.num_relax_jac_evals, r.delta_e));
    let retval = delta_e_fn(ark_mem, relax_jac_fn, &mut evals_out, &mut delta_e);
    relax_set(ark_mem, |r| {
        r.num_relax_jac_evals = evals_out;
        r.delta_e = delta_e;
    });
    if retval != 0 {
        return retval;
    }

    /* Get the change in state (delta_y = tempv2) */
    let (ycur, yn, tempv2) = {
        let m = ark_mem.borrow();
        (
            m.ycur.clone().expect("ycur"),
            m.yn.clone().expect("yn"),
            m.tempv2.clone().expect("tempv2"),
        )
    };
    N_VLinearSum(ONE, &ycur, -ONE, &yn, &tempv2);

    /* Store the current relaxation function value */
    let relax_fn = relax_get(ark_mem, |r| r.relax_fn.expect("relax_fn"));
    let mut e_old = relax_get(ark_mem, |r| r.e_old);
    let mut user_data = ark_mem.borrow_mut().user_data.take();
    let retval = relax_fn(&yn, &mut e_old, &mut user_data);
    ark_mem.borrow_mut().user_data = user_data;
    relax_set(ark_mem, |r| {
        r.e_old = e_old;
        r.num_relax_fn_evals += 1;
    });
    if retval < 0 {
        return ARK_RELAX_FUNC_FAIL;
    }
    if retval > 0 {
        return ARK_RELAX_FUNC_RECV;
    }

    /* Initial guess for relaxation parameter */
    relax_set(ark_mem, |r| r.relax_param = r.relax_param_prev);

    /* C's `default: return ARK_ILL_INPUT` arm is unreachable: ARKRelaxSolver
    has exactly these two variants */
    let retval = match relax_get(ark_mem, |r| r.solver) {
        ARK_RELAX_BRENT => arkRelaxBrentSolve(ark_mem),
        ARK_RELAX_NEWTON => arkRelaxNewtonSolve(ark_mem),
    };

    /* Check for solver failure */
    if retval != 0 {
        relax_set(ark_mem, |r| r.nls_fails += 1);
        return retval;
    }

    /* Check for bad relaxation value */
    let (relax_param, lower_bound, upper_bound) =
        relax_get(ark_mem, |r| (r.relax_param, r.lower_bound, r.upper_bound));
    if relax_param < lower_bound || relax_param > upper_bound {
        relax_set(ark_mem, |r| r.bound_fails += 1);
        return ARK_RELAX_SOLVE_RECV;
    }

    /* Save parameter for next initial guess */
    relax_set(ark_mem, |r| r.relax_param_prev = r.relax_param);

    /* Return relaxation value */
    *relax_val_out = relax_get(ark_mem, |r| r.relax_param);

    ARK_SUCCESS
}

/* =============================================================================
 * User Functions
 * ===========================================================================*/

/* -----------------------------------------------------------------------------
 * Set functions
 * ---------------------------------------------------------------------------*/

pub fn ARKodeSetRelaxFn(
    arkode_mem: &ARKodeMem,
    rfn: Option<ARKRelaxFn>,
    rjac: Option<ARKRelaxJacFn>,
) -> i32 {
    /* the C `arkode_mem == NULL` branch is unrepresentable */
    let ark_mem = arkode_mem;

    /* Ensure that the current N_Vector supports N_VDotProd */
    let tempv1 = ark_mem.borrow().tempv1.clone().expect("tempv1");
    if tempv1.ops.borrow().nvdotprod.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "ARKodeSetRelaxFn",
            file!(),
            "N_VDotProd unimplemented (required for relaxation)",
        );
        return ARK_ILL_INPUT;
    }

    /* Call stepper-specific routine (if it exists) */
    let step_setrelaxfn = ark_mem.borrow().step_setrelaxfn;
    if let Some(step_setrelaxfn) = step_setrelaxfn {
        step_setrelaxfn(ark_mem, rfn, rjac)
    } else {
        arkProcessError(
            Some(ark_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetRelaxFn",
            file!(),
            "time-stepping module does not support relaxation",
        );
        ARK_STEPPER_UNSUPPORTED
    }
}

pub fn ARKodeSetRelaxEtaFail(arkode_mem: &ARKodeMem, eta_fail: sunrealtype) -> i32 {
    let ark_mem = arkode_mem;

    let retval = arkRelaxAccessMem(ark_mem, "ARKodeSetRelaxEtaFail");
    if retval != 0 {
        return retval;
    }

    /* Guard against use for time steppers that do not allow relaxation */
    if !ark_mem.borrow().step_supports_relaxation {
        arkProcessError(
            Some(ark_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetRelaxEtaFail",
            file!(),
            "time-stepping module does not support relaxation",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    if eta_fail > ZERO && eta_fail < ONE {
        relax_set(ark_mem, |r| r.eta_fail = eta_fail);
    } else {
        relax_set(ark_mem, |r| r.eta_fail = ARK_RELAX_DEFAULT_ETA_FAIL);
    }

    ARK_SUCCESS
}

pub fn ARKodeSetRelaxLowerBound(arkode_mem: &ARKodeMem, lower: sunrealtype) -> i32 {
    let ark_mem = arkode_mem;

    let retval = arkRelaxAccessMem(ark_mem, "ARKodeSetRelaxLowerBound");
    if retval != 0 {
        return retval;
    }

    /* Guard against use for time steppers that do not allow relaxation */
    if !ark_mem.borrow().step_supports_relaxation {
        arkProcessError(
            Some(ark_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetRelaxLowerBound",
            file!(),
            "time-stepping module does not support relaxation",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    if lower > ZERO && lower < ONE {
        relax_set(ark_mem, |r| r.lower_bound = lower);
    } else {
        relax_set(ark_mem, |r| r.lower_bound = ARK_RELAX_DEFAULT_LOWER_BOUND);
    }

    ARK_SUCCESS
}

pub fn ARKodeSetRelaxMaxFails(arkode_mem: &ARKodeMem, max_fails: i32) -> i32 {
    let ark_mem = arkode_mem;

    let retval = arkRelaxAccessMem(ark_mem, "ARKodeSetRelaxMaxFails");
    if retval != 0 {
        return retval;
    }

    /* Guard against use for time steppers that do not allow relaxation */
    if !ark_mem.borrow().step_supports_relaxation {
        arkProcessError(
            Some(ark_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetRelaxMaxFails",
            file!(),
            "time-stepping module does not support relaxation",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    if max_fails > 0 {
        relax_set(ark_mem, |r| r.max_fails = max_fails);
    } else {
        relax_set(ark_mem, |r| r.max_fails = ARK_RELAX_DEFAULT_MAX_FAILS);
    }

    ARK_SUCCESS
}

pub fn ARKodeSetRelaxMaxIters(arkode_mem: &ARKodeMem, max_iters: i32) -> i32 {
    let ark_mem = arkode_mem;

    let retval = arkRelaxAccessMem(ark_mem, "ARKodeSetRelaxMaxIters");
    if retval != 0 {
        return retval;
    }

    /* Guard against use for time steppers that do not allow relaxation */
    if !ark_mem.borrow().step_supports_relaxation {
        arkProcessError(
            Some(ark_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetRelaxMaxIters",
            file!(),
            "time-stepping module does not support relaxation",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    if max_iters > 0 {
        relax_set(ark_mem, |r| r.max_iters = max_iters);
    } else {
        relax_set(ark_mem, |r| r.max_iters = ARK_RELAX_DEFAULT_MAX_ITERS);
    }

    ARK_SUCCESS
}

pub fn ARKodeSetRelaxSolver(arkode_mem: &ARKodeMem, solver: ARKRelaxSolver) -> i32 {
    let ark_mem = arkode_mem;

    let retval = arkRelaxAccessMem(ark_mem, "ARKodeSetRelaxSolver");
    if retval != 0 {
        return retval;
    }

    /* Guard against use for time steppers that do not allow relaxation */
    if !ark_mem.borrow().step_supports_relaxation {
        arkProcessError(
            Some(ark_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetRelaxSolver",
            file!(),
            "time-stepping module does not support relaxation",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    /* kept failure-path check (deviation class 1): unreachable because
    ARKRelaxSolver has exactly these two variants */
    if solver != ARK_RELAX_BRENT && solver != ARK_RELAX_NEWTON {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "ARKodeSetRelaxSolver",
            file!(),
            "An invalid relaxation solver option was provided.",
        );
        return ARK_ILL_INPUT;
    }

    relax_set(ark_mem, |r| r.solver = solver);

    ARK_SUCCESS
}

pub fn ARKodeSetRelaxResTol(arkode_mem: &ARKodeMem, res_tol: sunrealtype) -> i32 {
    let ark_mem = arkode_mem;

    let retval = arkRelaxAccessMem(ark_mem, "ARKodeSetRelaxResTol");
    if retval != 0 {
        return retval;
    }

    /* Guard against use for time steppers that do not allow relaxation */
    if !ark_mem.borrow().step_supports_relaxation {
        arkProcessError(
            Some(ark_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetRelaxResTol",
            file!(),
            "time-stepping module does not support relaxation",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    if res_tol > ZERO {
        relax_set(ark_mem, |r| r.res_tol = res_tol);
    } else {
        relax_set(ark_mem, |r| r.res_tol = ARK_RELAX_DEFAULT_RES_TOL);
    }

    ARK_SUCCESS
}

pub fn ARKodeSetRelaxTol(
    arkode_mem: &ARKodeMem,
    rel_tol: sunrealtype,
    abs_tol: sunrealtype,
) -> i32 {
    let ark_mem = arkode_mem;

    let retval = arkRelaxAccessMem(ark_mem, "ARKodeSetRelaxTol");
    if retval != 0 {
        return retval;
    }

    /* Guard against use for time steppers that do not allow relaxation */
    if !ark_mem.borrow().step_supports_relaxation {
        arkProcessError(
            Some(ark_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetRelaxTol",
            file!(),
            "time-stepping module does not support relaxation",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    if rel_tol > ZERO {
        relax_set(ark_mem, |r| r.rel_tol = rel_tol);
    } else {
        relax_set(ark_mem, |r| r.rel_tol = ARK_RELAX_DEFAULT_REL_TOL);
    }

    if abs_tol > ZERO {
        relax_set(ark_mem, |r| r.abs_tol = abs_tol);
    } else {
        relax_set(ark_mem, |r| r.abs_tol = ARK_RELAX_DEFAULT_ABS_TOL);
    }

    ARK_SUCCESS
}

pub fn ARKodeSetRelaxUpperBound(arkode_mem: &ARKodeMem, upper: sunrealtype) -> i32 {
    let ark_mem = arkode_mem;

    let retval = arkRelaxAccessMem(ark_mem, "ARKodeSetRelaxUpperBound");
    if retval != 0 {
        return retval;
    }

    /* Guard against use for time steppers that do not allow relaxation */
    if !ark_mem.borrow().step_supports_relaxation {
        arkProcessError(
            Some(ark_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetRelaxUpperBound",
            file!(),
            "time-stepping module does not support relaxation",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    if upper > ONE {
        relax_set(ark_mem, |r| r.upper_bound = upper);
    } else {
        relax_set(ark_mem, |r| r.upper_bound = ARK_RELAX_DEFAULT_UPPER_BOUND);
    }

    ARK_SUCCESS
}

/* -----------------------------------------------------------------------------
 * Get functions
 * ---------------------------------------------------------------------------*/

pub fn ARKodeGetNumRelaxFnEvals(arkode_mem: &ARKodeMem, r_evals: &mut i64) -> i32 {
    let ark_mem = arkode_mem;

    let retval = arkRelaxAccessMem(ark_mem, "ARKodeGetNumRelaxFnEvals");
    if retval != 0 {
        return retval;
    }

    /* Guard against use for time steppers that do not allow relaxation */
    if !ark_mem.borrow().step_supports_relaxation {
        arkProcessError(
            Some(ark_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeGetNumRelaxFnEvals",
            file!(),
            "time-stepping module does not support relaxation",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    *r_evals = relax_get(ark_mem, |r| r.num_relax_fn_evals);

    ARK_SUCCESS
}

pub fn ARKodeGetNumRelaxJacEvals(arkode_mem: &ARKodeMem, J_evals: &mut i64) -> i32 {
    let ark_mem = arkode_mem;

    let retval = arkRelaxAccessMem(ark_mem, "ARKodeGetNumRelaxJacEvals");
    if retval != 0 {
        return retval;
    }

    /* Guard against use for time steppers that do not allow relaxation */
    if !ark_mem.borrow().step_supports_relaxation {
        arkProcessError(
            Some(ark_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeGetNumRelaxJacEvals",
            file!(),
            "time-stepping module does not support relaxation",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    *J_evals = relax_get(ark_mem, |r| r.num_relax_jac_evals);

    ARK_SUCCESS
}

pub fn ARKodeGetNumRelaxFails(arkode_mem: &ARKodeMem, relax_fails: &mut i64) -> i32 {
    let ark_mem = arkode_mem;

    let retval = arkRelaxAccessMem(ark_mem, "ARKodeGetNumRelaxFails");
    if retval != 0 {
        return retval;
    }

    /* Guard against use for time steppers that do not allow relaxation */
    if !ark_mem.borrow().step_supports_relaxation {
        arkProcessError(
            Some(ark_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeGetNumRelaxFails",
            file!(),
            "time-stepping module does not support relaxation",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    *relax_fails = relax_get(ark_mem, |r| r.num_fails);

    ARK_SUCCESS
}

pub fn ARKodeGetNumRelaxSolveFails(arkode_mem: &ARKodeMem, fails: &mut i64) -> i32 {
    let ark_mem = arkode_mem;

    let retval = arkRelaxAccessMem(ark_mem, "ARKodeGetNumRelaxSolveFails");
    if retval != 0 {
        return retval;
    }

    /* Guard against use for time steppers that do not allow relaxation */
    if !ark_mem.borrow().step_supports_relaxation {
        arkProcessError(
            Some(ark_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeGetNumRelaxSolveFails",
            file!(),
            "time-stepping module does not support relaxation",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    *fails = relax_get(ark_mem, |r| r.nls_fails);

    ARK_SUCCESS
}

pub fn ARKodeGetNumRelaxBoundFails(arkode_mem: &ARKodeMem, fails: &mut i64) -> i32 {
    let ark_mem = arkode_mem;

    let retval = arkRelaxAccessMem(ark_mem, "ARKodeGetNumRelaxBoundFails");
    if retval != 0 {
        return retval;
    }

    /* Guard against use for time steppers that do not allow relaxation */
    if !ark_mem.borrow().step_supports_relaxation {
        arkProcessError(
            Some(ark_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeGetNumRelaxBoundFails",
            file!(),
            "time-stepping module does not support relaxation",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    *fails = relax_get(ark_mem, |r| r.bound_fails);

    ARK_SUCCESS
}

pub fn ARKodeGetNumRelaxSolveIters(arkode_mem: &ARKodeMem, iters: &mut i64) -> i32 {
    let ark_mem = arkode_mem;

    let retval = arkRelaxAccessMem(ark_mem, "ARKodeGetNumRelaxSolveIters");
    if retval != 0 {
        return retval;
    }

    /* Guard against use for time steppers that do not allow relaxation */
    if !ark_mem.borrow().step_supports_relaxation {
        arkProcessError(
            Some(ark_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeGetNumRelaxSolveIters",
            file!(),
            "time-stepping module does not support relaxation",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    *iters = relax_get(ark_mem, |r| r.nls_iters);

    ARK_SUCCESS
}

/* =============================================================================
 * Driver and Stepper Functions
 * ===========================================================================*/

/* Constructor called by stepper */
pub fn arkRelaxCreate(
    ark_mem: &ARKodeMem,
    relax_fn: Option<ARKRelaxFn>,
    relax_jac_fn: Option<ARKRelaxJacFn>,
    delta_e_fn: Option<ARKRelaxDeltaEFn>,
    get_order_fn: Option<ARKRelaxGetOrderFn>,
) -> i32 {
    /* Disable relaxation if both user inputs are NULL */
    if relax_fn.is_none() && relax_jac_fn.is_none() {
        ark_mem.borrow_mut().relax_enabled = SUNFALSE;
        return ARK_SUCCESS;
    }

    /* Ensure both the relaxation function and Jacobian are provided */
    if relax_fn.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "arkRelaxCreate",
            file!(),
            "The relaxation function is NULL.",
        );
        return ARK_ILL_INPUT;
    }

    if relax_jac_fn.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "arkRelaxCreate",
            file!(),
            "The relaxation Jacobian function is NULL.",
        );
        return ARK_ILL_INPUT;
    }

    /* Ensure stepper supplied inputs are provided */
    if delta_e_fn.is_none() || get_order_fn.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "arkRelaxCreate",
            file!(),
            "The Delta y, Delta e, or get order function is NULL.",
        );
        return ARK_ILL_INPUT;
    }

    /* Allocate and initialize relaxation memory structure (C `malloc` +
    `memset` to zero; allocation cannot fail here) */
    if ark_mem.borrow().relax_mem.is_none() {
        let mut m = ark_mem.borrow_mut();
        m.relax_mem = Some(Box::new(ARKodeRelaxMemRec {
            relax_fn: None,
            relax_jac_fn: None,
            delta_e_fn: None,
            get_order_fn: None,
            max_fails: 0,
            num_relax_fn_evals: 0,
            num_relax_jac_evals: 0,
            num_fails: 0,
            e_old: ZERO,
            delta_e: ZERO,
            res: ZERO,
            jac: ZERO,
            relax_param: ZERO,
            relax_param_prev: ZERO,
            lower_bound: ZERO,
            upper_bound: ZERO,
            eta_fail: ZERO,
            solver: ARK_RELAX_BRENT,
            res_tol: ZERO,
            rel_tol: ZERO,
            abs_tol: ZERO,
            max_iters: 0,
            nls_iters: 0,
            nls_fails: 0,
            bound_fails: 0,
        }));

        {
            let relax_mem = m.relax_mem.as_mut().expect("relax_mem");

            /* Set defaults */
            relax_mem.max_fails = ARK_RELAX_DEFAULT_MAX_FAILS;
            relax_mem.lower_bound = ARK_RELAX_DEFAULT_LOWER_BOUND;
            relax_mem.upper_bound = ARK_RELAX_DEFAULT_UPPER_BOUND;
            relax_mem.eta_fail = ARK_RELAX_DEFAULT_ETA_FAIL;
            relax_mem.solver = ARK_RELAX_NEWTON;
            relax_mem.res_tol = ARK_RELAX_DEFAULT_RES_TOL;
            relax_mem.rel_tol = ARK_RELAX_DEFAULT_REL_TOL;
            relax_mem.abs_tol = ARK_RELAX_DEFAULT_ABS_TOL;
            relax_mem.max_iters = ARK_RELAX_DEFAULT_MAX_ITERS;

            /* Initialize values */
            relax_mem.relax_param_prev = ONE;
        }

        /* Update workspace sizes */
        m.lrw += 12;
        m.liw += 14;
    }

    /* Set function pointers */
    relax_set(ark_mem, |r| {
        r.relax_fn = relax_fn;
        r.relax_jac_fn = relax_jac_fn;
        r.delta_e_fn = delta_e_fn;
        r.get_order_fn = get_order_fn;
    });

    /* Enable relaxation */
    ark_mem.borrow_mut().relax_enabled = SUNTRUE;

    ARK_SUCCESS
}

/* Destructor called by driver */
pub fn arkRelaxDestroy(relax_mem: Option<ARKodeRelaxMem>) -> i32 {
    if relax_mem.is_none() {
        return ARK_SUCCESS;
    }

    /* Free structure */
    drop(relax_mem);

    ARK_SUCCESS
}

/* Compute and apply relaxation, called by driver */
pub fn arkRelax(ark_mem: &ARKodeMem, relax_fails: &mut i32, dsm_inout: &mut sunrealtype) -> i32 {
    let mut relax_val: sunrealtype = ZERO;

    /* Get the relaxation memory structure */
    if ark_mem.borrow().relax_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_RELAX_MEM_NULL,
            line!() as i32,
            "arkRelax",
            file!(),
            MSG_RELAX_MEM_NULL,
        );
        return ARK_RELAX_MEM_NULL;
    }

    /* Compute the relaxation parameter */
    let retval = arkRelaxSolve(ark_mem, &mut relax_val);
    if retval < 0 {
        return retval;
    }
    if retval > 0 {
        /* Update failure counts */
        relax_set(ark_mem, |r| r.num_fails += 1);
        *relax_fails += 1;

        /* Check for max fails in a step */
        if *relax_fails == relax_get(ark_mem, |r| r.max_fails) {
            return ARK_RELAX_FAIL;
        }

        /* Return with an error if |h| == hmin */
        let (h, hmin) = {
            let m = ark_mem.borrow();
            (m.h, m.hmin)
        };
        if SUNRabs(h) <= hmin * ONEPSM {
            return ARK_RELAX_FAIL;
        }

        /* Return with error if using fixed step sizes */
        if ark_mem.borrow().fixedstep {
            return ARK_RELAX_FAIL;
        }

        /* Cut step size and try again */
        let eta_fail = relax_get(ark_mem, |r| r.eta_fail);
        ark_mem.borrow_mut().eta = eta_fail;

        return TRY_AGAIN;
    }

    /* Update step size and error estimate */
    ark_mem.borrow_mut().h *= relax_val;
    let get_order_fn = relax_get(ark_mem, |r| r.get_order_fn.expect("get_order_fn"));
    *dsm_inout *= SUNRpowerI(relax_val, get_order_fn(ark_mem));

    /* Relax solution */
    let (ycur, yn) = {
        let m = ark_mem.borrow();
        (m.ycur.clone().expect("ycur"), m.yn.clone().expect("yn"))
    };
    N_VLinearSum(relax_val, &ycur, ONE - relax_val, &yn, &ycur);

    ARK_SUCCESS
}

/* Print relaxation solver statistics, called by ARKODE */
pub fn arkRelaxPrintAllStats(
    arkode_mem: &ARKodeMem,
    outfile: &SUNFile,
    fmt: SUNOutputFormat,
) -> i32 {
    let ark_mem = arkode_mem;

    let retval = arkRelaxAccessMem(ark_mem, "arkRelaxPrintAllStats");
    if retval != 0 {
        return retval;
    }

    let (num_relax_fn_evals, num_relax_jac_evals, num_fails, bound_fails, nls_iters, nls_fails) =
        relax_get(ark_mem, |r| {
            (
                r.num_relax_fn_evals,
                r.num_relax_jac_evals,
                r.num_fails,
                r.bound_fails,
                r.nls_iters,
                r.nls_fails,
            )
        });

    sunfprintf_long(
        outfile,
        fmt,
        SUNFALSE,
        "Relax fn evals",
        num_relax_fn_evals,
    );
    sunfprintf_long(
        outfile,
        fmt,
        SUNFALSE,
        "Relax Jac evals",
        num_relax_jac_evals,
    );
    sunfprintf_long(outfile, fmt, SUNFALSE, "Relax fails", num_fails);
    sunfprintf_long(outfile, fmt, SUNFALSE, "Relax bound fails", bound_fails);
    sunfprintf_long(outfile, fmt, SUNFALSE, "Relax NLS iters", nls_iters);
    sunfprintf_long(outfile, fmt, SUNFALSE, "Relax NLS fails", nls_fails);

    ARK_SUCCESS
}

/* =============================================================================
 * EOF
 * ===========================================================================*/
