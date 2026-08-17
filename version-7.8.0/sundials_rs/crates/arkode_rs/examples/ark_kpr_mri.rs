/* ----------------------------------------------------------------
 * Rust port of examples/arkode/C_serial/ark_kpr_mri.c
 * Programmer(s): Daniel R. Reynolds @ UMBC
 *                Rujeko Chinomona @ UMBC
 * ----------------------------------------------------------------
 * Multirate nonlinear Kvaerno-Prothero-Robinson ODE test problem:
 *
 *    [u]' = [ G  e ] [(-1+u^2-r)/(2u)] + [      r'(t)/(2u)        ]
 *    [v]    [ e -1 ] [(-2+v^2-s)/(2v)]   [ s'(t)/(2*sqrt(2+s(t))) ]
 *         = [ fs(t,u,v) ]
 *           [ ff(t,u,v) ]
 *
 * where r(t) = 0.5*cos(t),  s(t) = cos(w*t),  0 < t < 5.
 *
 * This problem has analytical solution given by
 *    u(t) = sqrt(1+r(t)),  v(t) = sqrt(2+s(t)).
 *
 * We use the parameters:
 *   e = 0.5 (fast/slow coupling strength) [default]
 *   G = -1e2 (stiffness at slow time scale) [default]
 *   w = 100  (time-scale separation factor) [default]
 *   hs = 0.01 (slow step size) [default]
 *
 * The stiffness of the slow time scale is essentially determined
 * by G, for |G| > 50 it is 'stiff' and ideally suited to a
 * multirate method that is implicit at the slow time scale.
 *
 * We select the MRI method to use based on additional inputs:
 *
 *   slow_type:
 *      0 - none (full problem at fast scale)
 *      1 - ARKODE_MIS_KW3
 *      2 - ARKODE_MRI_GARK_ERK45a
 *      3 - ARKODE_MERK21
 *      4 - ARKODE_MERK32
 *      5 - ARKODE_MERK43
 *      6 - ARKODE_MERK54
 *      7 - ARKODE_MRI_GARK_IRK21a
 *      8 - ARKODE_MRI_GARK_ESDIRK34a
 *      9 - ARKODE_IMEX_MRI_GARK3b
 *     10 - ARKODE_IMEX_MRI_GARK4
 *     11 - ARKODE_IMEX_MRI_SR21
 *     12 - ARKODE_IMEX_MRI_SR32
 *     13 - ARKODE_IMEX_MRI_SR43
 *
 *   fast_type:
 *      0 - none (full problem at slow scale)
 *      1 - esdirk-3-3 (manually entered non-embedded table)
 *      2 - ARKODE_HEUN_EULER_2_1_2
 *      3 - erk-3-3 (manually entered non-embedded table)
 *      4 - erk-4-4 (manually entered non-embeded table)
 *      5 - ARKODE_DORMAND_PRINCE_7_4_5
 *
 * The program should be run with arguments in the following order:
 *   $ ark_kpr_mri slow_type fast_type h G w e deduce_rhs
 * Not all arguments are required, but these must be omitted from
 * end-to-beginning.
 *
 * This program solves the problem with the MRI stepper. Outputs are
 * printed at equal intervals of 0.1 and run statistics are printed
 * at the end.
 * ----------------------------------------------------------------*/

#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

use arkode_rs::prelude::*;

use std::any::Any;
use std::fs::File;
use std::io::Write;

const ZERO: sunrealtype = 0.0;
const ONE: sunrealtype = 1.0;
const TWO: sunrealtype = 2.0;

/* C macro `NV_Ith_S(v,i)` (0-based). The RefMut guard lives only for the
statement that uses it, per the workspace granular-borrow rule. */

fn NV_Ith_S(v: &N_Vector, i: usize) -> sunrealtype {
    NV_DATA_S(v)[i]
}

fn NV_Ith_S_set(v: &N_Vector, i: usize, x: sunrealtype) {
    NV_DATA_S(v)[i] = x;
}

/* Main Program */
fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let argc: i32 = argv.len() as i32;

    /* general problem parameters */
    let T0: sunrealtype = 0.0; /* initial time */
    let Tf: sunrealtype = 5.0; /* final time */
    let dTout: sunrealtype = 0.1; /* time between outputs */
    let NEQ: sunindextype = 2; /* number of dependent vars. */
    let Nt: i32 = (Tf / dTout).ceil() as i32; /* number of output times */
    let slow_type: i32; /* problem configuration type */
    let fast_type: i32; /* problem configuration type */
    let mut hs: sunrealtype = 0.01; /* slow step size */
    let mut e: sunrealtype = 0.5; /* fast/slow coupling strength */
    let mut G: sunrealtype = -100.0; /* stiffness at slow time scale */
    let mut w: sunrealtype = 100.0; /* time-scale separation factor */
    let mut reltol: sunrealtype = 0.01;
    let mut abstol: sunrealtype = 1.0e-11;

    /* general problem variables */
    let mut retval: i32; /* reusable error-checking flag */
    let mut inner_arkode_mem: Option<ARKodeMem> = None; /* ARKode memory structure */
    let mut inner_stepper: Option<MRIStepInnerStepper> = None; /* inner stepper */
    let B: Option<ARKodeButcherTable>; /* fast method Butcher table */
    let C: Option<MRIStepCoupling>; /* slow coupling coefficients */
    let mut Af: Option<SUNMatrix> = None; /* matrix for fast solver */
    let mut LSf: Option<SUNLinearSolver> = None; /* fast linear solver object */
    let mut As: Option<SUNMatrix> = None; /* matrix for slow solver */
    let mut LSs: Option<SUNLinearSolver> = None; /* slow linear solver object */
    let mut implicit_slow: sunbooleantype = false;
    let mut imex_slow: sunbooleantype = false;
    let mut explicit_slow: sunbooleantype = false;
    let mut no_slow: sunbooleantype = false;
    let mut implicit_fast: sunbooleantype = false;
    let mut explicit_fast: sunbooleantype = false;
    let mut no_fast: sunbooleantype = false;
    let mut deduce_rhs: sunbooleantype = false;
    let hf: sunrealtype;
    let gamma: sunrealtype;
    let beta: sunrealtype;
    let mut t: sunrealtype;
    let mut tout: sunrealtype;
    let mut rpar: [sunrealtype; 3] = [0.0; 3];
    let mut uerr: sunrealtype;
    let mut verr: sunrealtype;
    let mut uerrtot: sunrealtype;
    let mut verrtot: sunrealtype;
    let mut errtot: sunrealtype;
    let mut nsts: i64 = 0;
    let mut nstf: i64 = 0;
    let mut nfse: i64 = 0;
    let mut nfsi: i64 = 0;
    let mut nff: i64 = 0;
    let mut nnif: i64 = 0;
    let mut nncf: i64 = 0;
    let mut njef: i64 = 0;
    let mut nnis: i64 = 0;
    let mut nncs: i64 = 0;
    let mut njes: i64 = 0;

    /*
     * Initialization
     */

    /* Retrieve the command-line options: slow_type fast_type h G w e deduce_rhs */
    if argc < 3 {
        print!(
            "ERROR: executable requires at least two arguments [slow_type \
             fast_type]\n"
        );
        print!("Usage:\n");
        print!("  ark_kpr_mri slow_type fast_type h G w e deduce_rhs");
        std::process::exit(-1);
    }
    slow_type = argv[1].trim().parse().unwrap_or(0);
    fast_type = argv[2].trim().parse().unwrap_or(0);
    if argc > 3 {
        hs = SUNStrToReal(&argv[3]);
    }
    if argc > 4 {
        G = SUNStrToReal(&argv[4]);
    }
    if argc > 5 {
        w = SUNStrToReal(&argv[5]);
    }
    if argc > 6 {
        e = SUNStrToReal(&argv[6]);
    }
    if argc > 7 {
        deduce_rhs = argv[7].trim().parse::<i32>().unwrap_or(0) != 0;
    }

    /* Check arguments for validity */
    /*   0 <= slow_type <= 13      */
    /*   0 <= fast_type <= 5       */
    /*   G < 0.0                   */
    /*   h > 0                     */
    /*   h < 1/|G| (explicit slow) */
    /*   w >= 1.0                  */
    if (slow_type < 0) || (slow_type > 13) {
        print!("ERROR: slow_type be an integer in [0,13] \n");
        std::process::exit(-1);
    }
    if (fast_type < 0) || (fast_type > 5) {
        print!("ERROR: fast_type be an integer in [0,5] \n");
        std::process::exit(-1);
    }
    if (slow_type == 0) && (fast_type == 0) {
        print!("ERROR: at least one of slow_type and fast_type must be nonzero\n");
        std::process::exit(-1);
    }
    if (slow_type >= 9) && (fast_type == 0) {
        print!(
            "ERROR: example not configured for ImEx slow solver with no fast \
             solver\n"
        );
        std::process::exit(-1);
    }
    if G >= ZERO {
        print!("ERROR: G must be a negative real number\n");
        std::process::exit(-1);
    }
    if hs <= ZERO {
        print!("ERROR: hs must be in positive\n");
        std::process::exit(-1);
    }
    if (hs > ONE / SUNRabs(G)) && (!implicit_slow) {
        print!("ERROR: hs must be in (0, 1/|G|)\n");
        std::process::exit(-1);
    }
    if w < ONE {
        print!("ERROR: w must be >= 1.0\n");
        std::process::exit(-1);
    }
    rpar[0] = G;
    rpar[1] = w;
    rpar[2] = e;
    hf = hs / w;

    /* Initial problem output (and set implicit solver tolerances as needed) */
    print!("\nMultirate nonlinear Kvaerno-Prothero-Robinson test problem:\n");
    print!("    time domain:  ({},{}]\n", fmt_g(T0, 6), fmt_g(Tf, 6));
    print!("    hs = {}\n", fmt_g(hs, 6));
    print!("    hf = {}\n", fmt_g(hf, 6));
    print!("    G = {}\n", fmt_g(G, 6));
    print!("    w = {}\n", fmt_g(w, 6));
    print!("    e = {}\n", fmt_g(e, 6));
    match slow_type {
        0 => {
            print!("    slow solver: none\n");
            no_slow = true;
        }
        1 => {
            print!("    slow solver: ARKODE_MIS_KW3\n");
            explicit_slow = true;
        }
        2 => {
            print!("    slow solver: ARKODE_MRI_GARK_ERK45a\n");
            explicit_slow = true;
        }
        3 => {
            print!("    slow solver: ARKODE_MERK21\n");
            explicit_slow = true;
        }
        4 => {
            print!("    slow solver: ARKODE_MERK32\n");
            explicit_slow = true;
        }
        5 => {
            print!("    slow solver: ARKODE_MERK43\n");
            explicit_slow = true;
        }
        6 => {
            print!("    slow solver: ARKODE_MERK54\n");
            explicit_slow = true;
        }
        7 => {
            print!("    slow solver: ARKODE_MRI_GARK_IRK21a\n");
            implicit_slow = true;
            reltol = SUNMAX(hs * hs, 1.0e-10);
            abstol = 1.0e-11;
            print!(
                "      reltol = {},  abstol = {}\n",
                fmt_e(reltol, 2),
                fmt_e(abstol, 2)
            );
        }
        8 => {
            print!("    slow solver: ARKODE_MRI_GARK_ESDIRK34a\n");
            implicit_slow = true;
            reltol = SUNMAX(hs * hs * hs, 1.0e-10);
            abstol = 1.0e-11;
            print!(
                "      reltol = {},  abstol = {}\n",
                fmt_e(reltol, 2),
                fmt_e(abstol, 2)
            );
        }
        9 => {
            print!("    slow solver: ARKODE_IMEX_MRI_GARK3b\n");
            imex_slow = true;
            reltol = SUNMAX(hs * hs * hs, 1.0e-10);
            abstol = 1.0e-11;
            print!(
                "      reltol = {},  abstol = {}\n",
                fmt_e(reltol, 2),
                fmt_e(abstol, 2)
            );
        }
        10 => {
            print!("    slow solver: ARKODE_IMEX_MRI_GARK4\n");
            imex_slow = true;
            reltol = SUNMAX(hs * hs * hs * hs, 1.0e-14);
            abstol = 1.0e-14;
            print!(
                "      reltol = {},  abstol = {}\n",
                fmt_e(reltol, 2),
                fmt_e(abstol, 2)
            );
        }
        11 => {
            print!("    slow solver: ARKODE_IMEX_MRI_SR21\n");
            imex_slow = true;
            reltol = SUNMAX(hs * hs, 1.0e-10);
            abstol = 1.0e-11;
            print!(
                "      reltol = {},  abstol = {}\n",
                fmt_e(reltol, 2),
                fmt_e(abstol, 2)
            );
        }
        12 => {
            print!("    slow solver: ARKODE_IMEX_MRI_SR32\n");
            imex_slow = true;
            reltol = SUNMAX(hs * hs * hs, 1.0e-10);
            abstol = 1.0e-11;
            print!(
                "      reltol = {},  abstol = {}\n",
                fmt_e(reltol, 2),
                fmt_e(abstol, 2)
            );
        }
        13 => {
            print!("    slow solver: ARKODE_IMEX_MRI_SR43\n");
            imex_slow = true;
            reltol = SUNMAX(hs * hs * hs * hs, 1.0e-14);
            abstol = 1.0e-14;
            print!(
                "      reltol = {},  abstol = {}\n",
                fmt_e(reltol, 2),
                fmt_e(abstol, 2)
            );
        }
        _ => {}
    }
    match fast_type {
        0 => {
            print!("    fast solver: none\n");
            no_fast = true;
        }
        1 => {
            print!("    fast solver: esdirk-3-3\n");
            implicit_fast = true;
            reltol = SUNMAX(hs * hs * hs, 1.0e-10);
            abstol = 1.0e-11;
            print!(
                "      reltol = {},  abstol = {}\n",
                fmt_e(reltol, 2),
                fmt_e(abstol, 2)
            );
        }
        2 => {
            print!("    fast solver: ARKODE_HEUN_EULER_2_1_2\n");
            explicit_fast = true;
        }
        3 => {
            print!("    fast solver: erk-3-3\n");
            explicit_fast = true;
        }
        4 => {
            print!("    fast solver: erk-4-4\n");
            explicit_fast = true;
        }
        5 => {
            print!("    fast solver: ARKODE_DORMAND_PRINCE_7_4_5\n");
            explicit_fast = true;
        }
        _ => {}
    }

    /* Create the SUNDIALS context object for this simulation */
    let mut ctx: Option<SUNContext> = None;
    retval = SUNContext_Create(SUN_COMM_NULL, &mut ctx);
    if check_retval_int(retval, "SUNContext_Create") != 0 {
        std::process::exit(1);
    }
    let sunctx = ctx.clone().expect("SUNContext");

    /* Create and initialize serial vector for the solution */
    let y = N_VNew_Serial(NEQ, &sunctx);
    if check_retval_null(&y, "N_VNew_Serial") != 0 {
        std::process::exit(1);
    }
    let y = y.expect("N_VNew_Serial");
    retval = Ytrue(T0, &y, &rpar);
    if check_retval_int(retval, "Ytrue") != 0 {
        std::process::exit(1);
    }

    /*
     * Create the fast integrator and set options
     */

    /* Initialize the fast integrator. Specify the fast right-hand side
    function in y'=fs(t,y)+ff(t,y) = fse(t,y)+fsi(t,y)+ff(t,y), the initial time T0,
    and the initial dependent variable vector y.  If the fast scale is implicit,
    set up matrix, linear solver, and Jacobian function */
    if implicit_fast {
        Af = SUNDenseMatrix(NEQ, NEQ, &sunctx);
        if check_retval_null(&Af, "SUNDenseMatrix") != 0 {
            std::process::exit(1);
        }
        LSf = SUNLinSol_Dense(&y, Af.as_ref().expect("SUNDenseMatrix"), &sunctx);
        if check_retval_null(&LSf, "SUNLinSol_Dense") != 0 {
            std::process::exit(1);
        }
    }
    if no_fast {
        inner_arkode_mem = ARKStepCreate(Some(f0), None, T0, &y, &sunctx);
        if check_retval_null(&inner_arkode_mem, "ARKStepCreate") != 0 {
            std::process::exit(1);
        }
    } else if explicit_fast && !no_slow {
        inner_arkode_mem = ARKStepCreate(Some(ff), None, T0, &y, &sunctx);
        if check_retval_null(&inner_arkode_mem, "ARKStepCreate") != 0 {
            std::process::exit(1);
        }
    } else if explicit_fast && no_slow {
        inner_arkode_mem = ARKStepCreate(Some(fn_), None, T0, &y, &sunctx);
        if check_retval_null(&inner_arkode_mem, "ARKStepCreate") != 0 {
            std::process::exit(1);
        }
    } else if implicit_fast && no_slow {
        inner_arkode_mem = ARKStepCreate(None, Some(fn_), T0, &y, &sunctx);
        if check_retval_null(&inner_arkode_mem, "ARKStepCreate") != 0 {
            std::process::exit(1);
        }
        retval = ARKodeSetLinearSolver(
            inner_arkode_mem.as_ref().expect("ARKStepCreate"),
            LSf.as_ref().expect("SUNLinSol_Dense"),
            Af.as_ref(),
        );
        if check_retval_int(retval, "ARKodeSetLinearSolver") != 0 {
            std::process::exit(1);
        }
        retval = ARKodeSetJacFn(inner_arkode_mem.as_ref().expect("ARKStepCreate"), Some(Jn));
        if check_retval_int(retval, "ARKodeSetJacFn") != 0 {
            std::process::exit(1);
        }
    } else if implicit_fast && !no_slow {
        inner_arkode_mem = ARKStepCreate(None, Some(ff), T0, &y, &sunctx);
        if check_retval_null(&inner_arkode_mem, "ARKStepCreate") != 0 {
            std::process::exit(1);
        }
        retval = ARKodeSetLinearSolver(
            inner_arkode_mem.as_ref().expect("ARKStepCreate"),
            LSf.as_ref().expect("SUNLinSol_Dense"),
            Af.as_ref(),
        );
        if check_retval_int(retval, "ARKodeSetLinearSolver") != 0 {
            std::process::exit(1);
        }
        retval = ARKodeSetJacFn(inner_arkode_mem.as_ref().expect("ARKStepCreate"), Some(Jf));
        if check_retval_int(retval, "ARKodeSetJacFn") != 0 {
            std::process::exit(1);
        }
    }
    let inner_mem = inner_arkode_mem.clone().expect("ARKStepCreate");

    /* Set Butcher table for fast integrator */
    match fast_type {
        0 => {
            B = ARKodeButcherTable_Alloc(3, true);
            if check_retval_null(&B, "ARKodeButcherTable_Alloc") != 0 {
                std::process::exit(1);
            }
            {
                let Bt = B.as_ref().expect("ARKodeButcherTable_Alloc");
                let mut Bm = Bt.borrow_mut();
                Bm.A[1][0] = 0.5;
                Bm.A[2][0] = -ONE;
                Bm.A[2][1] = TWO;
                Bm.b[0] = ONE / 6.0;
                Bm.b[1] = TWO / 3.0;
                Bm.b[2] = ONE / 6.0;
                Bm.d[1] = ONE;
                Bm.c[1] = 0.5;
                Bm.c[2] = ONE;
                Bm.q = 3;
                Bm.p = 2;
            }
            retval = ARKStepSetTables(&inner_mem, 3, 2, None, B.as_ref());
            if check_retval_int(retval, "ARKStepSetTables") != 0 {
                std::process::exit(1);
            }
        }
        1 => {
            B = ARKodeButcherTable_Alloc(3, false);
            if check_retval_null(&B, "ARKodeButcherTable_Alloc") != 0 {
                std::process::exit(1);
            }
            beta = (3.0 as sunrealtype).sqrt() / 6.0 + 0.5;
            gamma = (-ONE / 8.0) * ((3.0 as sunrealtype).sqrt() + ONE);
            {
                let Bt = B.as_ref().expect("ARKodeButcherTable_Alloc");
                let mut Bm = Bt.borrow_mut();
                Bm.A[1][0] = 4.0 * gamma + TWO * beta;
                Bm.A[1][1] = ONE - 4.0 * gamma - TWO * beta;
                Bm.A[2][0] = 0.5 - beta - gamma;
                Bm.A[2][1] = gamma;
                Bm.A[2][2] = beta;
                Bm.b[0] = ONE / 6.0;
                Bm.b[1] = ONE / 6.0;
                Bm.b[2] = TWO / 3.0;
                Bm.c[1] = ONE;
                Bm.c[2] = 0.5;
                Bm.q = 3;
            }
            retval = ARKStepSetTables(&inner_mem, 3, 0, B.as_ref(), None);
            if check_retval_int(retval, "ARKStepSetTables") != 0 {
                std::process::exit(1);
            }
        }
        2 => {
            B = ARKodeButcherTable_LoadERK(ARKODE_HEUN_EULER_2_1_2);
            if check_retval_null(&B, "ARKodeButcherTable_LoadERK") != 0 {
                std::process::exit(1);
            }
            retval = ARKStepSetTables(&inner_mem, 2, 1, None, B.as_ref());
            if check_retval_int(retval, "ARKStepSetTables") != 0 {
                std::process::exit(1);
            }
        }
        3 => {
            B = ARKodeButcherTable_Alloc(3, true);
            if check_retval_null(&B, "ARKodeButcherTable_Alloc") != 0 {
                std::process::exit(1);
            }
            {
                let Bt = B.as_ref().expect("ARKodeButcherTable_Alloc");
                let mut Bm = Bt.borrow_mut();
                Bm.A[1][0] = 0.5;
                Bm.A[2][0] = -ONE;
                Bm.A[2][1] = TWO;
                Bm.b[0] = ONE / 6.0;
                Bm.b[1] = TWO / 3.0;
                Bm.b[2] = ONE / 6.0;
                Bm.d[1] = ONE;
                Bm.c[1] = 0.5;
                Bm.c[2] = ONE;
                Bm.q = 3;
                Bm.p = 2;
            }
            retval = ARKStepSetTables(&inner_mem, 3, 2, None, B.as_ref());
            if check_retval_int(retval, "ARKStepSetTables") != 0 {
                std::process::exit(1);
            }
        }
        4 => {
            B = ARKodeButcherTable_Alloc(4, false);
            if check_retval_null(&B, "ARKodeButcherTable_Alloc") != 0 {
                std::process::exit(1);
            }
            {
                let Bt = B.as_ref().expect("ARKodeButcherTable_Alloc");
                let mut Bm = Bt.borrow_mut();
                Bm.A[1][0] = 0.5;
                Bm.A[2][1] = 0.5;
                Bm.A[3][2] = ONE;
                Bm.b[0] = ONE / 6.0;
                Bm.b[1] = ONE / 3.0;
                Bm.b[2] = ONE / 3.0;
                Bm.b[3] = ONE / 6.0;
                Bm.c[1] = 0.5;
                Bm.c[2] = 0.5;
                Bm.c[3] = ONE;
                Bm.q = 4;
            }
            retval = ARKStepSetTables(&inner_mem, 4, 0, None, B.as_ref());
            if check_retval_int(retval, "ARKStepSetTables") != 0 {
                std::process::exit(1);
            }
        }
        5 => {
            B = ARKodeButcherTable_LoadERK(ARKODE_DORMAND_PRINCE_7_4_5);
            if check_retval_null(&B, "ARKodeButcherTable_LoadERK") != 0 {
                std::process::exit(1);
            }
            retval = ARKStepSetTables(&inner_mem, 5, 4, None, B.as_ref());
            if check_retval_int(retval, "ARKStepSetTables") != 0 {
                std::process::exit(1);
            }
        }
        _ => {
            B = None;
        }
    }
    ARKodeButcherTable_Free(B);

    /* Set the tolerances */
    retval = ARKodeSStolerances(&inner_mem, reltol, abstol);
    if check_retval_int(retval, "ARKodeSStolerances") != 0 {
        std::process::exit(1);
    }

    /* Set the user data pointer */
    retval = ARKodeSetUserData(&inner_mem, Some(Box::new(rpar)));
    if check_retval_int(retval, "ARKodeSetUserData") != 0 {
        std::process::exit(1);
    }

    /* Set the fast step size */
    retval = ARKodeSetFixedStep(&inner_mem, hf);
    if check_retval_int(retval, "ARKodeSetFixedStep") != 0 {
        std::process::exit(1);
    }

    /* Override any current settings with command-line options -- enforce
    the prefix "inner" */
    retval = ARKodeSetOptions(&inner_mem, Some("inner"), Some(""), argc, &argv);
    if check_retval_int(retval, "ARKodeSetOptions") != 0 {
        std::process::exit(1);
    }

    /* Create inner stepper */
    retval = ARKodeCreateMRIStepInnerStepper(&inner_mem, &mut inner_stepper);
    if check_retval_int(retval, "ARKodeCreateMRIStepInnerStepper") != 0 {
        std::process::exit(1);
    }
    let stepper = inner_stepper.clone().expect("MRIStepInnerStepper");

    /*
     * Create the slow integrator and set options
     */

    /* Initialize the slow integrator. Specify the slow right-hand side
    function in y'=fs(t,y)+ff(t,y) = fse(t,y)+fsi(t,y)+ff(t,y), the initial time
    T0, the initial dependent variable vector y, and the fast integrator.  If
    the slow scale contains an implicit component, set up matrix, linear solver,
    and Jacobian function. */
    let mut arkode_mem: Option<ARKodeMem> = None;
    if implicit_slow || imex_slow {
        As = SUNDenseMatrix(NEQ, NEQ, &sunctx);
        if check_retval_null(&As, "SUNDenseMatrix") != 0 {
            std::process::exit(1);
        }
        LSs = SUNLinSol_Dense(&y, As.as_ref().expect("SUNDenseMatrix"), &sunctx);
        if check_retval_null(&LSs, "SUNLinSol_Dense") != 0 {
            std::process::exit(1);
        }
    }
    if no_slow {
        arkode_mem = MRIStepCreate(Some(f0), None, T0, &y, &stepper, &sunctx);
        if check_retval_null(&arkode_mem, "MRIStepCreate") != 0 {
            std::process::exit(1);
        }
    } else if explicit_slow && !no_fast {
        arkode_mem = MRIStepCreate(Some(fs), None, T0, &y, &stepper, &sunctx);
        if check_retval_null(&arkode_mem, "MRIStepCreate") != 0 {
            std::process::exit(1);
        }
    } else if explicit_slow && no_fast {
        arkode_mem = MRIStepCreate(Some(fn_), None, T0, &y, &stepper, &sunctx);
        if check_retval_null(&arkode_mem, "MRIStepCreate") != 0 {
            std::process::exit(1);
        }
    } else if implicit_slow && !no_fast {
        arkode_mem = MRIStepCreate(None, Some(fs), T0, &y, &stepper, &sunctx);
        if check_retval_null(&arkode_mem, "MRIStepCreate") != 0 {
            std::process::exit(1);
        }
        retval = ARKodeSetLinearSolver(
            arkode_mem.as_ref().expect("MRIStepCreate"),
            LSs.as_ref().expect("SUNLinSol_Dense"),
            As.as_ref(),
        );
        if check_retval_int(retval, "ARKodeSetLinearSolver") != 0 {
            std::process::exit(1);
        }
        retval = ARKodeSetJacFn(arkode_mem.as_ref().expect("MRIStepCreate"), Some(Js));
        if check_retval_int(retval, "ARKodeSetJacFn") != 0 {
            std::process::exit(1);
        }
    } else if implicit_slow && no_fast {
        arkode_mem = MRIStepCreate(None, Some(fn_), T0, &y, &stepper, &sunctx);
        if check_retval_null(&arkode_mem, "MRIStepCreate") != 0 {
            std::process::exit(1);
        }
        retval = ARKodeSetLinearSolver(
            arkode_mem.as_ref().expect("MRIStepCreate"),
            LSs.as_ref().expect("SUNLinSol_Dense"),
            As.as_ref(),
        );
        if check_retval_int(retval, "ARKodeSetLinearSolver") != 0 {
            std::process::exit(1);
        }
        retval = ARKodeSetJacFn(arkode_mem.as_ref().expect("MRIStepCreate"), Some(Jn));
        if check_retval_int(retval, "ARKodeSetJacFn") != 0 {
            std::process::exit(1);
        }
    } else if imex_slow {
        arkode_mem = MRIStepCreate(Some(fse), Some(fsi), T0, &y, &stepper, &sunctx);
        if check_retval_null(&arkode_mem, "MRIStepCreate") != 0 {
            std::process::exit(1);
        }
        retval = ARKodeSetLinearSolver(
            arkode_mem.as_ref().expect("MRIStepCreate"),
            LSs.as_ref().expect("SUNLinSol_Dense"),
            As.as_ref(),
        );
        if check_retval_int(retval, "ARKodeSetLinearSolver") != 0 {
            std::process::exit(1);
        }
        retval = ARKodeSetJacFn(arkode_mem.as_ref().expect("MRIStepCreate"), Some(Jsi));
        if check_retval_int(retval, "ARKodeSetJacFn") != 0 {
            std::process::exit(1);
        }
    }
    let mri_mem = arkode_mem.clone().expect("MRIStepCreate");

    /* Set coupling table for slow integrator */
    match slow_type {
        0 => {
            /* no slow dynamics (use ERK-2-2) */
            let Bs = ARKodeButcherTable_Alloc(2, false);
            if check_retval_null(&Bs, "ARKodeButcherTable_Alloc") != 0 {
                std::process::exit(1);
            }
            {
                let Bt = Bs.as_ref().expect("ARKodeButcherTable_Alloc");
                let mut Bm = Bt.borrow_mut();
                Bm.A[1][0] = TWO / 3.0;
                Bm.b[0] = 0.25;
                Bm.b[1] = 0.75;
                Bm.c[1] = TWO / 3.0;
                Bm.q = 2;
            }
            C = MRIStepCoupling_MIStoMRI(Bs.as_ref(), 2, 0);
            if check_retval_null(&C, "MRIStepCoupling_MIStoMRI") != 0 {
                std::process::exit(1);
            }
            ARKodeButcherTable_Free(Bs);
        }
        1 => {
            C = MRIStepCoupling_LoadTable(ARKODE_MIS_KW3);
            if check_retval_null(&C, "MRIStepCoupling_LoadTable") != 0 {
                std::process::exit(1);
            }
        }
        2 => {
            C = MRIStepCoupling_LoadTable(ARKODE_MRI_GARK_ERK45a);
            if check_retval_coupling_deref(&C, "MRIStepCoupling_LoadTable") != 0 {
                std::process::exit(1);
            }
        }
        3 => {
            C = MRIStepCoupling_LoadTable(ARKODE_MERK21);
            if check_retval_coupling_deref(&C, "MRIStepCoupling_LoadTable") != 0 {
                std::process::exit(1);
            }
        }
        4 => {
            C = MRIStepCoupling_LoadTable(ARKODE_MERK32);
            if check_retval_coupling_deref(&C, "MRIStepCoupling_LoadTable") != 0 {
                std::process::exit(1);
            }
        }
        5 => {
            C = MRIStepCoupling_LoadTable(ARKODE_MERK43);
            if check_retval_coupling_deref(&C, "MRIStepCoupling_LoadTable") != 0 {
                std::process::exit(1);
            }
        }
        6 => {
            C = MRIStepCoupling_LoadTable(ARKODE_MERK54);
            if check_retval_coupling_deref(&C, "MRIStepCoupling_LoadTable") != 0 {
                std::process::exit(1);
            }
        }
        7 => {
            C = MRIStepCoupling_LoadTable(ARKODE_MRI_GARK_IRK21a);
            if check_retval_coupling_deref(&C, "MRIStepCoupling_LoadTable") != 0 {
                std::process::exit(1);
            }
        }
        8 => {
            C = MRIStepCoupling_LoadTable(ARKODE_MRI_GARK_ESDIRK34a);
            if check_retval_coupling_deref(&C, "MRIStepCoupling_LoadTable") != 0 {
                std::process::exit(1);
            }
        }
        9 => {
            C = MRIStepCoupling_LoadTable(ARKODE_IMEX_MRI_GARK3b);
            if check_retval_null(&C, "MRIStepCoupling_LoadTable") != 0 {
                std::process::exit(1);
            }
        }
        10 => {
            C = MRIStepCoupling_LoadTable(ARKODE_IMEX_MRI_GARK4);
            if check_retval_null(&C, "MRIStepCoupling_LoadTable") != 0 {
                std::process::exit(1);
            }
        }
        11 => {
            C = MRIStepCoupling_LoadTable(ARKODE_IMEX_MRI_SR21);
            if check_retval_null(&C, "MRIStepCoupling_LoadTable") != 0 {
                std::process::exit(1);
            }
        }
        12 => {
            C = MRIStepCoupling_LoadTable(ARKODE_IMEX_MRI_SR32);
            if check_retval_null(&C, "MRIStepCoupling_LoadTable") != 0 {
                std::process::exit(1);
            }
        }
        13 => {
            C = MRIStepCoupling_LoadTable(ARKODE_IMEX_MRI_SR43);
            if check_retval_null(&C, "MRIStepCoupling_LoadTable") != 0 {
                std::process::exit(1);
            }
        }
        _ => {
            C = None;
        }
    }
    retval = MRIStepSetCoupling(&mri_mem, C.as_ref().expect("MRIStepCoupling"));
    if check_retval_int(retval, "MRIStepSetCoupling") != 0 {
        std::process::exit(1);
    }
    MRIStepCoupling_Free(C); /* free coupling coefficients */

    /* Set the tolerances */
    retval = ARKodeSStolerances(&mri_mem, reltol, abstol);
    if check_retval_int(retval, "ARKodeSStolerances") != 0 {
        std::process::exit(1);
    }

    /* Set the user data pointer */
    retval = ARKodeSetUserData(&mri_mem, Some(Box::new(rpar)));
    if check_retval_int(retval, "ARKodeSetUserData") != 0 {
        std::process::exit(1);
    }

    retval = ARKodeSetDeduceImplicitRhs(&mri_mem, deduce_rhs);
    if check_retval_int(retval, "ARKodeSetDeduceImplicitRhs") != 0 {
        std::process::exit(1);
    }

    /* Set the slow step size */
    retval = ARKodeSetFixedStep(&mri_mem, hs);
    if check_retval_int(retval, "ARKodeSetFixedStep") != 0 {
        std::process::exit(1);
    }

    /* Override any current settings with command-line options -- enforce
    the prefix "outer" */
    retval = ARKodeSetOptions(&mri_mem, Some("outer"), Some(""), argc, &argv);
    if check_retval_int(retval, "ARKodeSetOptions") != 0 {
        std::process::exit(1);
    }

    /*
     * Integrate ODE
     */

    /* Open output stream for results, output comment line */
    let mut UFID = File::create("ark_kpr_mri_solution.txt").expect("output file");
    let _ = write!(UFID, "# t u v uerr verr\n");

    /* output initial condition to disk */
    let _ = write!(
        UFID,
        " {} {} {} {} {}\n",
        fmt_e(T0, 16),
        fmt_e(NV_Ith_S(&y, 0), 16),
        fmt_e(NV_Ith_S(&y, 1), 16),
        fmt_e(SUNRabs(NV_Ith_S(&y, 0) - utrue(T0, &rpar)), 16),
        fmt_e(SUNRabs(NV_Ith_S(&y, 1) - vtrue(T0, &rpar)), 16)
    );

    /* Main time-stepping loop: calls ARKodeEvolve to perform the
    integration, then prints results. Stops when the final time
    has been reached */
    t = T0;
    tout = T0 + dTout;
    uerr = ZERO;
    verr = ZERO;
    uerrtot = ZERO;
    verrtot = ZERO;
    errtot = ZERO;
    print!("        t           u           v       uerr      verr\n");
    print!("   ------------------------------------------------------\n");
    print!(
        "  {}  {}  {}  {}  {}\n",
        fmt_fw(t, 10, 6),
        fmt_fw(NV_Ith_S(&y, 0), 10, 6),
        fmt_fw(NV_Ith_S(&y, 1), 10, 6),
        fmt_e(uerr, 2),
        fmt_e(verr, 2)
    );

    for _iout in 0..Nt {
        /* call integrator */
        retval = ARKodeEvolve(&mri_mem, tout, &y, &mut t, ARK_NORMAL);
        if check_retval_int(retval, "ARKodeEvolve") != 0 {
            break;
        }

        /* access/print solution and error */
        uerr = SUNRabs(NV_Ith_S(&y, 0) - utrue(t, &rpar));
        verr = SUNRabs(NV_Ith_S(&y, 1) - vtrue(t, &rpar));
        print!(
            "  {}  {}  {}  {}  {}\n",
            fmt_fw(t, 10, 6),
            fmt_fw(NV_Ith_S(&y, 0), 10, 6),
            fmt_fw(NV_Ith_S(&y, 1), 10, 6),
            fmt_e(uerr, 2),
            fmt_e(verr, 2)
        );
        let _ = write!(
            UFID,
            " {} {} {} {} {}\n",
            fmt_e(t, 16),
            fmt_e(NV_Ith_S(&y, 0), 16),
            fmt_e(NV_Ith_S(&y, 1), 16),
            fmt_e(uerr, 16),
            fmt_e(verr, 16)
        );
        uerrtot += uerr * uerr;
        verrtot += verr * verr;
        errtot += uerr * uerr + verr * verr;

        /* successful solve: update time */
        tout += dTout;
        tout = if tout > Tf { Tf } else { tout };
    }
    uerrtot = (uerrtot / Nt as sunrealtype).sqrt();
    verrtot = (verrtot / Nt as sunrealtype).sqrt();
    errtot = (errtot / Nt as sunrealtype / 2.0).sqrt();
    print!("   ------------------------------------------------------\n");
    drop(UFID);

    /*
     * Finalize
     */

    /* Get some slow integrator statistics */
    retval = ARKodeGetNumSteps(&mri_mem, &mut nsts);
    check_retval_int(retval, "ARKodeGetNumSteps");
    retval = ARKodeGetNumRhsEvals(&mri_mem, 0, &mut nfse);
    check_retval_int(retval, "ARKodeGetNumRhsEvals");
    retval = ARKodeGetNumRhsEvals(&mri_mem, 1, &mut nfsi);
    check_retval_int(retval, "ARKodeGetNumRhsEvals");

    /* Get some fast integrator statistics */
    retval = ARKodeGetNumSteps(&inner_mem, &mut nstf);
    check_retval_int(retval, "ARKodeGetNumSteps");
    retval = ARKodeGetNumRhsEvals(&inner_mem, 0, &mut nff);
    check_retval_int(retval, "ARKodeGetNumRhsEvals");

    /* Print some final statistics */
    print!("\nFinal Solver Statistics:\n");
    print!("   Steps: nsts = {}, nstf = {}\n", nsts, nstf);
    print!(
        "   u error = {}, v error = {}, total error = {}\n",
        fmt_e(uerrtot, 3),
        fmt_e(verrtot, 3),
        fmt_e(errtot, 3)
    );
    if imex_slow {
        print!(
            "   Total RHS evals:  Fse = {}, Fsi = {},  Ff = {}\n",
            nfse, nfsi, nff
        );
    } else if implicit_slow {
        print!("   Total RHS evals:  Fs = {},  Ff = {}\n", nfsi, nff);
    } else {
        print!("   Total RHS evals:  Fs = {},  Ff = {}\n", nfse, nff);
    }

    /* Get/print slow integrator decoupled implicit solver statistics */
    if implicit_slow || imex_slow {
        retval = ARKodeGetNonlinSolvStats(&mri_mem, &mut nnis, &mut nncs);
        check_retval_int(retval, "ARKodeGetNonlinSolvStats");
        retval = ARKodeGetNumJacEvals(&mri_mem, &mut njes);
        check_retval_int(retval, "ARKodeGetNumJacEvals");
        print!("   Slow Newton iters = {}\n", nnis);
        print!("   Slow Newton conv fails = {}\n", nncs);
        print!("   Slow Jacobian evals = {}\n", njes);
    }

    /* Get/print fast integrator implicit solver statistics */
    if implicit_fast {
        retval = ARKodeGetNonlinSolvStats(&inner_mem, &mut nnif, &mut nncf);
        check_retval_int(retval, "ARKodeGetNonlinSolvStats");
        retval = ARKodeGetNumJacEvals(&inner_mem, &mut njef);
        check_retval_int(retval, "ARKodeGetNumJacEvals");
        print!("   Fast Newton iters = {}\n", nnif);
        print!("   Fast Newton conv fails = {}\n", nncf);
        print!("   Fast Jacobian evals = {}\n", njef);
    }

    /* Clean up and return */
    N_VDestroy(y); /* Free y vector */
    if let Some(Af) = Af {
        SUNMatDestroy(Af); /* free fast matrix */
    }
    let _ = SUNLinSolFree(LSf); /* free fast linear solver */
    if let Some(As) = As {
        SUNMatDestroy(As); /* free fast matrix */
    }
    let _ = SUNLinSolFree(LSs); /* free fast linear solver */
    ARKodeFree(&mut inner_arkode_mem); /* Free fast integrator memory */
    let _ = MRIStepInnerStepper_Free(&mut inner_stepper); /* Free inner stepper */
    ARKodeFree(&mut arkode_mem); /* Free slow integrator memory */
    let _ = SUNContext_Free(&mut ctx); /* Free context */
}

/* ------------------------------
 * Functions called by the solver
 * ------------------------------*/

/* Downcast helper: the C `void* user_data` is the `rpar[3]` array. */
fn rpar_of(user_data: &mut Option<Box<dyn Any>>) -> [sunrealtype; 3] {
    *user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<[sunrealtype; 3]>())
        .expect("user_data")
}

/* ff routine to compute the fast portion of the ODE RHS. */
fn ff(t: sunrealtype, y: &N_Vector, ydot: &N_Vector, user_data: &mut Option<Box<dyn Any>>) -> i32 {
    let rpar = rpar_of(user_data);
    let e: sunrealtype = rpar[2];
    let u: sunrealtype = NV_Ith_S(y, 0);
    let v: sunrealtype = NV_Ith_S(y, 1);
    let tmp1: sunrealtype;
    let tmp2: sunrealtype;

    /* fill in the RHS function:
    [0  0]*[(-1+u^2-r(t))/(2*u)] + [         0          ]
    [e -1] [(-2+v^2-s(t))/(2*v)]   [sdot(t)/(2*vtrue(t))] */
    tmp1 = (-ONE + u * u - r(t, &rpar)) / (TWO * u);
    tmp2 = (-TWO + v * v - s(t, &rpar)) / (TWO * v);
    NV_Ith_S_set(ydot, 0, ZERO);
    NV_Ith_S_set(
        ydot,
        1,
        e * tmp1 - tmp2 + sdot(t, &rpar) / (TWO * vtrue(t, &rpar)),
    );

    /* Return with success */
    0
}

/* fs routine to compute the slow portion of the ODE RHS. */
fn fs(t: sunrealtype, y: &N_Vector, ydot: &N_Vector, user_data: &mut Option<Box<dyn Any>>) -> i32 {
    let rpar = rpar_of(user_data);
    let G: sunrealtype = rpar[0];
    let e: sunrealtype = rpar[2];
    let u: sunrealtype = NV_Ith_S(y, 0);
    let v: sunrealtype = NV_Ith_S(y, 1);
    let tmp1: sunrealtype;
    let tmp2: sunrealtype;

    /* fill in the RHS function:
    [G e]*[(-1+u^2-r(t))/(2*u))] + [rdot(t)/(2*u)]
    [0 0] [(-2+v^2-s(t))/(2*v)]    [      0      ] */
    tmp1 = (-ONE + u * u - r(t, &rpar)) / (TWO * u);
    tmp2 = (-TWO + v * v - s(t, &rpar)) / (TWO * v);
    NV_Ith_S_set(ydot, 0, G * tmp1 + e * tmp2 + rdot(t, &rpar) / (TWO * u));
    NV_Ith_S_set(ydot, 1, ZERO);

    /* Return with success */
    0
}

/* fse routine to compute the slow portion of the ODE RHS. */
fn fse(t: sunrealtype, y: &N_Vector, ydot: &N_Vector, user_data: &mut Option<Box<dyn Any>>) -> i32 {
    let rpar = rpar_of(user_data);
    let u: sunrealtype = NV_Ith_S(y, 0);

    /* fill in the slow explicit RHS function:
    [rdot(t)/(2*u)]
    [      0      ] */
    NV_Ith_S_set(ydot, 0, rdot(t, &rpar) / (TWO * u));
    NV_Ith_S_set(ydot, 1, ZERO);

    /* Return with success */
    0
}

/* fsi routine to compute the slow portion of the ODE RHS.(currently same as fse) */
fn fsi(t: sunrealtype, y: &N_Vector, ydot: &N_Vector, user_data: &mut Option<Box<dyn Any>>) -> i32 {
    let rpar = rpar_of(user_data);
    let G: sunrealtype = rpar[0];
    let e: sunrealtype = rpar[2];
    let u: sunrealtype = NV_Ith_S(y, 0);
    let v: sunrealtype = NV_Ith_S(y, 1);
    let tmp1: sunrealtype;
    let tmp2: sunrealtype;

    /* fill in the slow implicit RHS function:
    [G e]*[(-1+u^2-r(t))/(2*u))]
    [0 0] [(-2+v^2-s(t))/(2*v)]  */
    tmp1 = (-ONE + u * u - r(t, &rpar)) / (TWO * u);
    tmp2 = (-TWO + v * v - s(t, &rpar)) / (TWO * v);
    NV_Ith_S_set(ydot, 0, G * tmp1 + e * tmp2);
    NV_Ith_S_set(ydot, 1, ZERO);

    /* Return with success */
    0
}

/* C `fn` (renamed: `fn` is a Rust keyword) */
fn fn_(t: sunrealtype, y: &N_Vector, ydot: &N_Vector, user_data: &mut Option<Box<dyn Any>>) -> i32 {
    let rpar = rpar_of(user_data);
    let G: sunrealtype = rpar[0];
    let e: sunrealtype = rpar[2];
    let u: sunrealtype = NV_Ith_S(y, 0);
    let v: sunrealtype = NV_Ith_S(y, 1);
    let tmp1: sunrealtype;
    let tmp2: sunrealtype;

    /* fill in the RHS function:
    [G e]*[(-1+u^2-r(t))/(2*u))] + [rdot(t)/(2*u)]
    [e -1] [(-2+v^2-s(t))/(2*v)]   [sdot(t)/(2*vtrue(t))] */
    tmp1 = (-ONE + u * u - r(t, &rpar)) / (TWO * u);
    tmp2 = (-TWO + v * v - s(t, &rpar)) / (TWO * v);
    NV_Ith_S_set(ydot, 0, G * tmp1 + e * tmp2 + rdot(t, &rpar) / (TWO * u));
    NV_Ith_S_set(
        ydot,
        1,
        e * tmp1 - tmp2 + sdot(t, &rpar) / (TWO * vtrue(t, &rpar)),
    );

    /* Return with success */
    0
}

fn f0(
    _t: sunrealtype,
    _y: &N_Vector,
    ydot: &N_Vector,
    _user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    N_VConst(ZERO, ydot);
    0
}

fn Js(
    t: sunrealtype,
    y: &N_Vector,
    _fy: &N_Vector,
    J: &SUNMatrix,
    user_data: &mut Option<Box<dyn Any>>,
    _tmp1: &N_Vector,
    _tmp2: &N_Vector,
    _tmp3: &N_Vector,
) -> i32 {
    let rpar = rpar_of(user_data);
    let G: sunrealtype = rpar[0];
    let e: sunrealtype = rpar[2];
    let u: sunrealtype = NV_Ith_S(y, 0);
    let v: sunrealtype = NV_Ith_S(y, 1);

    /* fill in the Jacobian:
    [G/2 + (G*(1+r(t))-rdot(t))/(2*u^2)   e/2+e*(2+s(t))/(2*v^2)]
    [                 0                             0           ] */
    SM_ELEMENT_D_set(
        J,
        0,
        0,
        G / TWO + (G * (ONE + r(t, &rpar)) - rdot(t, &rpar)) / (2.0 * u * u),
    );
    SM_ELEMENT_D_set(J, 0, 1, e / TWO + e * (TWO + s(t, &rpar)) / (TWO * v * v));
    SM_ELEMENT_D_set(J, 1, 0, ZERO);
    SM_ELEMENT_D_set(J, 1, 1, ZERO);

    /* Return with success */
    0
}

fn Jsi(
    t: sunrealtype,
    y: &N_Vector,
    _fy: &N_Vector,
    J: &SUNMatrix,
    user_data: &mut Option<Box<dyn Any>>,
    _tmp1: &N_Vector,
    _tmp2: &N_Vector,
    _tmp3: &N_Vector,
) -> i32 {
    let rpar = rpar_of(user_data);
    let G: sunrealtype = rpar[0];
    let e: sunrealtype = rpar[2];
    let u: sunrealtype = NV_Ith_S(y, 0);
    let v: sunrealtype = NV_Ith_S(y, 1);

    /* fill in the Jacobian:
    [G/2 + (G*(1+r(t)))/(2*u^2)   e/2 + e*(2+s(t))/(2*v^2)]
    [                 0                       0           ] */
    SM_ELEMENT_D_set(J, 0, 0, G / TWO + (G * (ONE + r(t, &rpar))) / (2.0 * u * u));
    SM_ELEMENT_D_set(J, 0, 1, e / TWO + e * (TWO + s(t, &rpar)) / (TWO * v * v));
    SM_ELEMENT_D_set(J, 1, 0, ZERO);
    SM_ELEMENT_D_set(J, 1, 1, ZERO);

    /* Return with success */
    0
}

fn Jn(
    t: sunrealtype,
    y: &N_Vector,
    _fy: &N_Vector,
    J: &SUNMatrix,
    user_data: &mut Option<Box<dyn Any>>,
    _tmp1: &N_Vector,
    _tmp2: &N_Vector,
    _tmp3: &N_Vector,
) -> i32 {
    let rpar = rpar_of(user_data);
    let G: sunrealtype = rpar[0];
    let e: sunrealtype = rpar[2];
    let u: sunrealtype = NV_Ith_S(y, 0);
    let v: sunrealtype = NV_Ith_S(y, 1);

    /* fill in the Jacobian:
    [G/2 + (G*(1+r(t))-rdot(t))/(2*u^2)     e/2 + e*(2+s(t))/(2*v^2)]
    [e/2+e*(1+r(t))/(2*u^2)                -1/2 - (2+s(t))/(2*v^2)  ] */
    SM_ELEMENT_D_set(
        J,
        0,
        0,
        G / TWO + (G * (ONE + r(t, &rpar)) - rdot(t, &rpar)) / (2.0 * u * u),
    );
    SM_ELEMENT_D_set(J, 0, 1, e / TWO + e * (TWO + s(t, &rpar)) / (TWO * v * v));
    SM_ELEMENT_D_set(J, 1, 0, e / TWO + e * (ONE + r(t, &rpar)) / (TWO * u * u));
    SM_ELEMENT_D_set(J, 1, 1, -ONE / TWO - (TWO + s(t, &rpar)) / (TWO * v * v));

    /* Return with success */
    0
}

fn Jf(
    t: sunrealtype,
    y: &N_Vector,
    _fy: &N_Vector,
    J: &SUNMatrix,
    user_data: &mut Option<Box<dyn Any>>,
    _tmp1: &N_Vector,
    _tmp2: &N_Vector,
    _tmp3: &N_Vector,
) -> i32 {
    let rpar = rpar_of(user_data);
    let e: sunrealtype = rpar[2];
    let u: sunrealtype = NV_Ith_S(y, 0);
    let v: sunrealtype = NV_Ith_S(y, 1);

    /* fill in the Jacobian:
    [        0                           0        ]
    [e/2+e*(1+r(t))/(2*u^2)  -1/2-(2+s(t))/(2*v^2)] */
    SM_ELEMENT_D_set(J, 0, 0, ZERO);
    SM_ELEMENT_D_set(J, 0, 1, ZERO);
    SM_ELEMENT_D_set(J, 1, 0, e / TWO + e * (ONE + r(t, &rpar)) / (TWO * u * u));
    SM_ELEMENT_D_set(J, 1, 1, -ONE / TWO - (TWO + s(t, &rpar)) / (TWO * v * v));

    /* Return with success */
    0
}

/* ------------------------------
 * Private helper functions
 * ------------------------------*/

fn r(t: sunrealtype, _user_data: &[sunrealtype; 3]) -> sunrealtype {
    0.5 * t.sun_cos()
}

fn s(t: sunrealtype, user_data: &[sunrealtype; 3]) -> sunrealtype {
    let rpar = user_data;
    (rpar[1] * t).sun_cos()
}

fn rdot(t: sunrealtype, _user_data: &[sunrealtype; 3]) -> sunrealtype {
    -0.5 * t.sun_sin()
}

fn sdot(t: sunrealtype, user_data: &[sunrealtype; 3]) -> sunrealtype {
    let rpar = user_data;
    -rpar[1] * (rpar[1] * t).sun_sin()
}

fn utrue(t: sunrealtype, user_data: &[sunrealtype; 3]) -> sunrealtype {
    (ONE + r(t, user_data)).sqrt()
}

fn vtrue(t: sunrealtype, user_data: &[sunrealtype; 3]) -> sunrealtype {
    (TWO + s(t, user_data)).sqrt()
}

fn Ytrue(t: sunrealtype, y: &N_Vector, user_data: &[sunrealtype; 3]) -> i32 {
    NV_Ith_S_set(y, 0, utrue(t, user_data));
    NV_Ith_S_set(y, 1, vtrue(t, user_data));
    0
}

/* Check function return value...
opt == 0 means SUNDIALS function allocates memory so check if
         returned NULL pointer
opt == 1 means SUNDIALS function returns a retval so check if
         retval < 0
opt == 2 means function allocates memory so check if returned
         NULL pointer
*/
fn check_retval_null<T>(returnvalue: &Option<T>, funcname: &str) -> i32 {
    if returnvalue.is_none() {
        eprint!(
            "\nSUNDIALS_ERROR: {}() failed - returned NULL pointer\n\n",
            funcname
        );
        return 1;
    }
    0
}

fn check_retval_int(retval: i32, funcname: &str) -> i32 {
    if retval < 0 {
        eprint!(
            "\nSUNDIALS_ERROR: {}() failed with retval = {}\n\n",
            funcname, retval
        );
        return 1;
    }
    0
}

/* Upstream passes the `MRIStepCoupling` HANDLE to `check_retval` with
`opt == 1` for slow_type 2..8, so C reinterprets it as `int*` and tests the
leading `enum MRISTEP_METHOD_TYPE type` field of `MRIStepCouplingMem` (always
>= 0 for a loaded table, so the check never fires).  A NULL table would be a
NULL dereference in C — accepted deviation class 5 (C UB -> deterministic
panic). */
fn check_retval_coupling_deref(C: &Option<MRIStepCoupling>, funcname: &str) -> i32 {
    let retval = C
        .as_ref()
        .expect("NULL MRIStepCoupling dereference")
        .borrow()
        .type_ as i32;
    if retval < 0 {
        eprint!(
            "\nSUNDIALS_ERROR: {}() failed with retval = {}\n\n",
            funcname, retval
        );
        return 1;
    }
    0
}

/*---- end of file ----*/
