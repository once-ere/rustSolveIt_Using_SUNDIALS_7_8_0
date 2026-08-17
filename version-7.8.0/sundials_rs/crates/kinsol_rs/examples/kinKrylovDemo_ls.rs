/* -----------------------------------------------------------------
 * Rust port of examples/kinsol/serial/kinKrylovDemo_ls.c
 * -----------------------------------------------------------------
 *
 * This example loops through the available iterative linear solvers:
 * SPGMR, SPBCGS, SPTFQMR, and SPFGMR.
 *
 * Example (serial):
 *
 * This example solves a nonlinear system that arises from a system
 * of partial differential equations. The PDE system is a food web
 * population model, with predator-prey interaction and diffusion
 * on the unit square in two dimensions. The dependent variable
 * vector is the following:
 *
 *       1   2         ns
 * c = (c , c ,  ..., c  )     (denoted by the variable cc)
 *
 * and the PDE's are as follows:
 *
 *                    i       i
 *         0 = d(i)*(c     + c    )  +  f  (x,y,c)   (i=1,...,ns)
 *                    xx      yy         i
 *
 *   where
 *
 *                   i             ns         j
 *   f  (x,y,c)  =  c  * (b(i)  + sum a(i,j)*c )
 *    i                           j=1
 *
 * The number of species is ns = 2 * np, with the first np being
 * prey and the last np being predators. The number np is both the
 * number of prey and predator species. The coefficients a(i,j),
 * b(i), d(i) are:
 *
 *   a(i,i) = -AA   (all i)
 *   a(i,j) = -GG   (i <= np , j >  np)
 *   a(i,j) =  EE   (i >  np,  j <= np)
 *   b(i) = BB * (1 + alpha * x * y)   (i <= np)
 *   b(i) =-BB * (1 + alpha * x * y)   (i >  np)
 *   d(i) = DPREY   (i <= np)
 *   d(i) = DPRED   ( i > np)
 *
 * The various scalar parameters are set using define's or in
 * routine InitUserData.
 *
 * The boundary conditions are: normal derivative = 0, and the
 * initial guess is constant in x and y, but the final solution
 * is not.
 *
 * The PDEs are discretized by central differencing on an MX by
 * MY mesh.
 *
 * The nonlinear system is solved by KINSOL using the method
 * specified in local variable globalstrat.
 *
 * The preconditioner matrix is a block-diagonal matrix based on
 * the partial derivatives of the interaction terms f only.
 *
 * Constraints are imposed to make all components of the solution
 * positive.
 * -----------------------------------------------------------------
 */

#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

use std::any::Any;

use kinsol_rs::prelude::*;
use kinsol_rs::sundials_dense::{SUNDlsMat_denseGETRF, SUNDlsMat_denseGETRS};

/* helpful macros */

fn MAX(A: sunrealtype, B: sunrealtype) -> sunrealtype {
    if A > B {
        A
    } else {
        B
    }
}

/* Problem Constants */

const NUM_SPECIES: i32 = 6; /* must equal 2*(number of prey or predators)
                            number of prey = number of predators       */

const MX: i32 = 5; /* MX = number of x mesh points */
const MY: i32 = 5; /* MY = number of y mesh points */
const NSMX: i32 = NUM_SPECIES * MX;
const NEQ: i32 = NSMX * MY; /* number of equations in the system */
const AA: sunrealtype = 1.0; /* value of coefficient AA in above eqns */
const EE: sunrealtype = 10000.; /* value of coefficient EE in above eqns */
const GG: sunrealtype = 0.5e-6; /* value of coefficient GG in above eqns */
const BB: sunrealtype = 1.0; /* value of coefficient BB in above eqns */
const DPREY: sunrealtype = 1.0; /* value of coefficient dprey above */
const DPRED: sunrealtype = 0.5; /* value of coefficient dpred above */
const ALPHA: sunrealtype = 1.0; /* value of coefficient alpha above */
const AX: sunrealtype = 1.0; /* total range of x variable */
const AY: sunrealtype = 1.0; /* total range of y variable */
const FTOL: sunrealtype = 1.0e-7; /* ftol tolerance */
const STOL: sunrealtype = 1.0e-13; /* stol tolerance */
const THOUSAND: sunrealtype = 1000.0; /* one thousand */
const ZERO: sunrealtype = 0.; /* 0. */
const ONE: sunrealtype = 1.0; /* 1. */
const TWO: sunrealtype = 2.0; /* 2. */
const PREYIN: sunrealtype = 1.0; /* initial guess for prey concentrations. */
const PREDIN: sunrealtype = 30000.0; /* initial guess for predator concs.      */

/* Linear Solver Loop Constants */

const USE_SPGMR: i32 = 0;
const USE_SPBCGS: i32 = 1;
const USE_SPTFQMR: i32 = 2;
const USE_SPFGMR: i32 = 3;

/* User-defined vector access helper: IJ_Vptr */

/* IJ_Vptr is defined in order to translate from the underlying 3D structure
of the dependent variable vector to the 1D storage scheme for an N-vector.
IJ_Vptr(i,j) returns the flat offset in the vector data corresponding to
indices is = 0, jx = i, jy = j.    */

fn IJ_Vptr(i: i32, j: i32) -> usize {
    (i * NUM_SPECIES + j * NSMX) as usize
}

/* Type : UserData
contains preconditioner blocks, pivot arrays, and problem constants.

C `sunrealtype** P[MX][MY]` (a column-pointer dense block per mesh point)
maps to `P[jx][jy][col][row]`; `sunindextype* pivot[MX][MY]` maps to
`pivot[jx][jy][k]`; `sunrealtype** acoef` maps to `acoef[i][j]` with the
same index meaning the C code uses. */

struct UserData {
    P: Vec<Vec<Vec<Vec<sunrealtype>>>>,
    pivot: Vec<Vec<Vec<sunindextype>>>,
    acoef: Vec<Vec<sunrealtype>>,
    bcoef: Vec<sunrealtype>,
    rates: Option<N_Vector>,
    cox: Vec<sunrealtype>,
    coy: Vec<sunrealtype>,
    ax: sunrealtype,
    ay: sunrealtype,
    dx: sunrealtype,
    dy: sunrealtype,
    uround: sunrealtype,
    sqruround: sunrealtype,
    /* Kept for fidelity with the C struct; never read after init. */
    #[allow(dead_code)]
    mx: i32,
    #[allow(dead_code)]
    my: i32,
    #[allow(dead_code)]
    ns: i32,
    np: i32,
}

/*
 *--------------------------------------------------------------------
 * MAIN PROGRAM
 *--------------------------------------------------------------------
 */

fn main() {
    let globalstrategy: i32;
    let fnormtol: sunrealtype;
    let scsteptol: sunrealtype;
    let mut flag: i32;
    let mut maxl: i32 = 0;
    let mut maxlrst: i32 = 0;
    let mut LS: Option<SUNLinearSolver> = None;

    /* Create the SUNDIALS context object for this simulation. */
    let mut sunctx_opt: Option<SUNContext> = None;
    SUNContext_Create(SUN_COMM_NULL, &mut sunctx_opt);
    let sunctx = sunctx_opt.clone().expect("SUNContext_Create");

    /* Allocate memory, and set problem data, initial values, tolerances */
    globalstrategy = KIN_NONE;

    let data = AllocUserData();
    if check_flag_ptr(&data, "AllocUserData", 2) != 0 {
        std::process::exit(1);
    }
    let mut data = data.expect("AllocUserData");
    InitUserData(&mut data);

    /* Create serial vectors of length NEQ */
    let cc = N_VNew_Serial(NEQ as sunindextype, &sunctx);
    if check_flag_ptr(&cc, "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    let cc = cc.expect("N_VNew_Serial");
    let sc = N_VNew_Serial(NEQ as sunindextype, &sunctx);
    if check_flag_ptr(&sc, "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    let sc = sc.expect("N_VNew_Serial");
    let rates = N_VNew_Serial(NEQ as sunindextype, &sunctx);
    if check_flag_ptr(&rates, "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    data.rates = rates;

    let constraints = N_VNew_Serial(NEQ as sunindextype, &sunctx);
    if check_flag_ptr(&constraints, "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    let constraints = constraints.expect("N_VNew_Serial");
    N_VConst(TWO, &constraints);

    fnormtol = FTOL;
    scsteptol = STOL;

    /* The user data block travels from one KINSOL object to the next; it is
    swapped back out of the solver memory (KINGetUserData) before each
    KINFree so the following pass can hand it to a fresh KINSOL object. */
    let mut data_any: Option<Box<dyn Any>> = Some(Box::new(data));

    /* START: Loop through SPGMR, SPBCGS, SPTFQMR and SPFGMR linear solver modules */
    for linsolver in 0..4 {
        /* (Re-)Initialize user data */
        SetInitialProfiles(&cc, &sc);

        /* Call KINCreate/KINInit to initialize KINSOL:
        A pointer to KINSOL problem memory is returned and stored in kmem. */
        let kmem = KINCreate(&sunctx);
        if check_flag_ptr(&kmem, "KINCreate", 0) != 0 {
            std::process::exit(1);
        }
        let mut kmem_opt = kmem;
        let kmem = kmem_opt.as_ref().expect("KINCreate").clone();

        /* Vector cc passed as template vector. */
        flag = KINInit(&kmem, func, &cc);
        if check_flag(flag, "KINInit", 1) != 0 {
            std::process::exit(1);
        }

        flag = KINSetUserData(&kmem, data_any.take());
        if check_flag(flag, "KINSetUserData", 1) != 0 {
            std::process::exit(1);
        }
        flag = KINSetConstraints(&kmem, Some(&constraints));
        if check_flag(flag, "KINSetConstraints", 1) != 0 {
            std::process::exit(1);
        }
        flag = KINSetFuncNormTol(&kmem, fnormtol);
        if check_flag(flag, "KINSetFuncNormTol", 1) != 0 {
            std::process::exit(1);
        }
        flag = KINSetScaledStepTol(&kmem, scsteptol);
        if check_flag(flag, "KINSetScaledStepTol", 1) != 0 {
            std::process::exit(1);
        }

        /* Attach a linear solver module */
        match linsolver {
            /* (a) SPGMR */
            USE_SPGMR => {
                /* Print header */
                print!(" -------");
                print!(" \n| SPGMR |\n");
                print!(" -------\n");

                /* Create SUNLinSol_SPGMR object with right preconditioning and the
                maximum Krylov dimension maxl */
                maxl = 15;
                LS = SUNLinSol_SPGMR(&cc, SUN_PREC_RIGHT, maxl, &sunctx);
                if check_flag_ptr(&LS, "SUNLinSol_SPGMR", 0) != 0 {
                    std::process::exit(1);
                }

                /* Attach the linear solver to KINSOL */
                flag = KINSetLinearSolver(&kmem, LS.as_ref().expect("LS"), None);
                if check_flag(flag, "KINSetLinearSolver", 1) != 0 {
                    std::process::exit(1);
                }

                /* Set the maximum number of restarts */
                maxlrst = 2;
                flag = SUNLinSol_SPGMRSetMaxRestarts(LS.as_ref().expect("LS"), maxlrst);
                if check_flag(flag, "SUNLinSol_SPGMRSetMaxRestarts", 1) != 0 {
                    std::process::exit(1);
                }
            }

            /* (b) SPBCGS */
            USE_SPBCGS => {
                /* Print header */
                print!(" --------");
                print!(" \n| SPBCGS |\n");
                print!(" --------\n");

                /* Create SUNLinSol_SPBCGS object with right preconditioning and the
                maximum Krylov dimension maxl */
                maxl = 15;
                LS = SUNLinSol_SPBCGS(&cc, SUN_PREC_RIGHT, maxl, &sunctx);
                if check_flag_ptr(&LS, "SUNLinSol_SPBCGS", 0) != 0 {
                    std::process::exit(1);
                }

                /* Attach the linear solver to KINSOL */
                flag = KINSetLinearSolver(&kmem, LS.as_ref().expect("LS"), None);
                if check_flag(flag, "KINSetLinearSolver", 1) != 0 {
                    std::process::exit(1);
                }
            }

            /* (c) SPTFQMR */
            USE_SPTFQMR => {
                /* Print header */
                print!(" ---------");
                print!(" \n| SPTFQMR |\n");
                print!(" ---------\n");

                /* Create SUNLinSol_SPTFQMR object with right preconditioning and the
                maximum Krylov dimension maxl */
                maxl = 25;
                LS = SUNLinSol_SPTFQMR(&cc, SUN_PREC_RIGHT, maxl, &sunctx);
                if check_flag_ptr(&LS, "SUNLinSol_SPTFQMR", 0) != 0 {
                    std::process::exit(1);
                }

                /* Attach the linear solver to KINSOL */
                flag = KINSetLinearSolver(&kmem, LS.as_ref().expect("LS"), None);
                if check_flag(flag, "KINSetLinearSolver", 1) != 0 {
                    std::process::exit(1);
                }
            }

            /* (d) SPFGMR */
            USE_SPFGMR => {
                /* Print header */
                print!(" -------");
                print!(" \n| SPFGMR |\n");
                print!(" -------\n");

                /* Create SUNLinSol_SPFGMR object with right preconditioning and the
                maximum Krylov dimension maxl */
                maxl = 15;
                LS = SUNLinSol_SPFGMR(&cc, SUN_PREC_RIGHT, maxl, &sunctx);
                if check_flag_ptr(&LS, "SUNLinSol_SPFGMR", 0) != 0 {
                    std::process::exit(1);
                }

                /* Attach the linear solver to KINSOL */
                flag = KINSetLinearSolver(&kmem, LS.as_ref().expect("LS"), None);
                if check_flag(flag, "KINSetLinearSolver", 1) != 0 {
                    std::process::exit(1);
                }

                /* Set the maximum number of restarts.

                The C source calls SUNLinSol_SPGMRSetMaxRestarts() on the
                SPFGMR object; in C that writes the `max_restarts` field of
                the SPFGMR content struct, whose leading members
                (maxl, pretype, gstype, max_restarts) are laid out
                identically to SPGMR's. The typed Rust port calls the SPFGMR
                setter, which has exactly the same effect; the checked name
                below is kept as it appears in the C source. */
                maxlrst = 2;
                flag = SUNLinSol_SPFGMRSetMaxRestarts(LS.as_ref().expect("LS"), maxlrst);
                if check_flag(flag, "SUNLinSol_SPGMRSetMaxRestarts", 1) != 0 {
                    std::process::exit(1);
                }
            }

            _ => {}
        }

        /* Set preconditioner functions */
        flag = KINSetPreconditioner(&kmem, Some(PrecSetupBD), Some(PrecSolveBD));
        if check_flag(flag, "KINSetPreconditioner", 1) != 0 {
            std::process::exit(1);
        }

        /* Print out the problem size, solution parameters, initial guess. */
        PrintHeader(
            globalstrategy,
            maxl,
            maxlrst,
            fnormtol,
            scsteptol,
            linsolver,
        );

        /* Call KINSol and print output concentration profile */
        flag = KINSol(
            &kmem,          /* KINSol memory block */
            &cc,            /* initial guess on input; solution vector */
            globalstrategy, /* global strategy choice */
            &sc,            /* scaling vector, for the variable cc */
            &sc,
        ); /* scaling vector for function values fval */
        if check_flag(flag, "KINSol", 1) != 0 {
            std::process::exit(1);
        }

        print!("\n\nComputed equilibrium species concentrations:\n");
        PrintOutput(&cc);

        /* Print final statistics and free memory */
        PrintFinalStats(&kmem, linsolver);

        /* Take the user data box back before KINFree drops it. */
        KINGetUserData(&kmem, &mut data_any);

        drop(kmem);
        KINFree(&mut kmem_opt);
        SUNLinSolFree(LS.take());
    } /* END: Loop through SPGMR, SPBCGS, SPTFQMR, and SPFGMR linear solver modules */

    N_VDestroy(constraints);
    N_VDestroy(cc);
    N_VDestroy(sc);
    /* FreeUserData: dropping the box releases the preconditioner blocks,
    the pivot arrays and the rates vector. */
    drop(data_any);

    drop(sunctx);
    SUNContext_Free(&mut sunctx_opt);
}

/*
 *--------------------------------------------------------------------
 * FUNCTIONS CALLED BY KINSOL
 *--------------------------------------------------------------------
 */

/*
 * System function for predator-prey system
 */

fn func(cc: &N_Vector, fval: &N_Vector, user_data: &mut Option<Box<dyn Any>>) -> i32 {
    let data = user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<UserData>())
        .expect("user_data downcast");
    let delx = data.dx;
    let dely = data.dy;

    let rates = data.rates.clone().expect("data->rates");
    let ccdata = N_VGetArrayPointer(cc).expect("N_VGetArrayPointer");
    let mut rdata = N_VGetArrayPointer(&rates).expect("N_VGetArrayPointer");
    let mut fdata = N_VGetArrayPointer(fval).expect("N_VGetArrayPointer");

    /* Loop over all mesh points, evaluating rate array at each point*/
    for jy in 0..MY {
        let yy = dely * jy as sunrealtype;

        /* Set lower/upper index shifts, special at boundaries. */
        let idyl: isize = if jy != 0 {
            NSMX as isize
        } else {
            -(NSMX as isize)
        };
        let idyu: isize = if jy != MY - 1 {
            NSMX as isize
        } else {
            -(NSMX as isize)
        };

        for jx in 0..MX {
            let xx = delx * jx as sunrealtype;

            /* Set left/right index shifts, special at boundaries. */
            let idxl: isize = if jx != 0 {
                NUM_SPECIES as isize
            } else {
                -(NUM_SPECIES as isize)
            };
            let idxr: isize = if jx != MX - 1 {
                NUM_SPECIES as isize
            } else {
                -(NUM_SPECIES as isize)
            };

            let cxy = IJ_Vptr(jx, jy);
            let rxy = IJ_Vptr(jx, jy);
            let fxy = IJ_Vptr(jx, jy);

            /* Get species interaction rate array at (xx,yy) */
            WebRate(
                xx,
                yy,
                &ccdata[cxy..cxy + NUM_SPECIES as usize],
                &mut rdata[rxy..rxy + NUM_SPECIES as usize],
                data,
            );

            let c = cxy as isize;
            for is in 0..NUM_SPECIES as isize {
                /* Differencing in x direction */
                let dcyli = ccdata[(c + is) as usize] - ccdata[(c - idyl + is) as usize];
                let dcyui = ccdata[(c + idyu + is) as usize] - ccdata[(c + is) as usize];

                /* Differencing in y direction */
                let dcxli = ccdata[(c + is) as usize] - ccdata[(c - idxl + is) as usize];
                let dcxri = ccdata[(c + idxr + is) as usize] - ccdata[(c + is) as usize];

                /* Compute the total rate value at (xx,yy) */
                fdata[fxy + is as usize] = data.coy[is as usize] * (dcyui - dcyli)
                    + data.cox[is as usize] * (dcxri - dcxli)
                    + rdata[rxy + is as usize];
            } /* end of is loop */
        } /* end of jx loop */
    } /* end of jy loop */

    0
}

/*
 * Preconditioner setup routine. Generate and preprocess P.
 */

fn PrecSetupBD(
    cc: &N_Vector,
    cscale: &N_Vector,
    fval: &N_Vector,
    fscale: &N_Vector,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let mut perturb_rates = [ZERO; NUM_SPECIES as usize];

    let data = user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<UserData>())
        .expect("user_data downcast");
    let delx = data.dx;
    let dely = data.dy;

    let uround = data.uround;
    let sqruround = data.sqruround;
    let mut fac = N_VWL2Norm(fval, fscale);
    let mut r0 = THOUSAND * uround * fac * NEQ as sunrealtype;
    if r0 == ZERO {
        r0 = ONE;
    }

    let rates = data.rates.clone().expect("data->rates");
    let mut ccdata = N_VGetArrayPointer(cc).expect("N_VGetArrayPointer");
    let csdata = N_VGetArrayPointer(cscale).expect("N_VGetArrayPointer");
    let rdata = N_VGetArrayPointer(&rates).expect("N_VGetArrayPointer");

    /* Loop over spatial points; get size NUM_SPECIES Jacobian block at each */
    for jy in 0..MY {
        let yy = jy as sunrealtype * dely;

        for jx in 0..MX {
            let xx = jx as sunrealtype * delx;
            let cxy = IJ_Vptr(jx, jy);
            let scxy = IJ_Vptr(jx, jy);
            let ratesxy = IJ_Vptr(jx, jy);

            /* Compute difference quotients of interaction rate fn. */
            for j in 0..NUM_SPECIES as usize {
                let csave = ccdata[cxy + j]; /* Save the j,jx,jy element of cc */
                let r = MAX(sqruround * SUNRabs(csave), r0 / csdata[scxy + j]);
                ccdata[cxy + j] += r; /* Perturb the j,jx,jy element of cc */
                fac = ONE / r;

                WebRate(
                    xx,
                    yy,
                    &ccdata[cxy..cxy + NUM_SPECIES as usize],
                    &mut perturb_rates,
                    data,
                );

                /* Restore j,jx,jy element of cc */
                ccdata[cxy + j] = csave;

                /* Load the j-th column of difference quotients */
                for i in 0..NUM_SPECIES as usize {
                    data.P[jx as usize][jy as usize][j][i] =
                        (perturb_rates[i] - rdata[ratesxy + i]) * fac;
                }
            } /* end of j loop */

            /* Do LU decomposition of size NUM_SPECIES preconditioner block */
            let d = &mut *data;
            let mut Pxy: Vec<&mut [sunrealtype]> = d.P[jx as usize][jy as usize]
                .iter_mut()
                .map(|col| col.as_mut_slice())
                .collect();
            let ret = SUNDlsMat_denseGETRF(
                &mut Pxy,
                NUM_SPECIES as sunindextype,
                NUM_SPECIES as sunindextype,
                &mut d.pivot[jx as usize][jy as usize],
            );
            if ret != 0 {
                return 1;
            }
        } /* end of jx loop */
    } /* end of jy loop */

    0
}

/*
 * Preconditioner solve routine
 */

fn PrecSolveBD(
    _cc: &N_Vector,
    _cscale: &N_Vector,
    _fval: &N_Vector,
    _fscale: &N_Vector,
    vv: &N_Vector,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let data = user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<UserData>())
        .expect("user_data downcast");

    let mut vdata = N_VGetArrayPointer(vv).expect("N_VGetArrayPointer");

    for jx in 0..MX {
        for jy in 0..MY {
            /* For each (jx,jy), solve a linear system of size NUM_SPECIES.
            vxy is the offset of the corresponding portion of the vector vv;
            Pxy is the corresponding block of the matrix P;
            piv is the corresponding block of the array pivot. */
            let vxy = IJ_Vptr(jx, jy);
            let d = &mut *data;
            let mut Pxy: Vec<&mut [sunrealtype]> = d.P[jx as usize][jy as usize]
                .iter_mut()
                .map(|col| col.as_mut_slice())
                .collect();
            let piv = &d.pivot[jx as usize][jy as usize];
            SUNDlsMat_denseGETRS(
                &mut Pxy,
                NUM_SPECIES as sunindextype,
                piv,
                &mut vdata[vxy..vxy + NUM_SPECIES as usize],
            );
        } /* end of jy loop */
    } /* end of jx loop */

    0
}

/*
 * Interaction rate function routine
 */

fn WebRate(
    xx: sunrealtype,
    yy: sunrealtype,
    cxy: &[sunrealtype],
    ratesxy: &mut [sunrealtype],
    data: &UserData,
) {
    for i in 0..NUM_SPECIES as usize {
        ratesxy[i] = DotProd(NUM_SPECIES, cxy, &data.acoef[i]);
    }

    let fac = ONE + ALPHA * xx * yy;

    for i in 0..NUM_SPECIES as usize {
        ratesxy[i] = cxy[i] * (data.bcoef[i] * fac + ratesxy[i]);
    }
}

/*
 * Dot product routine for sunrealtype arrays
 */

fn DotProd(size: i32, x1: &[sunrealtype], x2: &[sunrealtype]) -> sunrealtype {
    let mut temp: sunrealtype = ZERO;

    for i in 0..size as usize {
        temp += x1[i] * x2[i];
    }

    temp
}

/*
 *--------------------------------------------------------------------
 * PRIVATE FUNCTIONS
 *--------------------------------------------------------------------
 */

/*
 * Allocate memory for data structure of type UserData
 */

fn AllocUserData() -> Option<UserData> {
    let ns = NUM_SPECIES as usize;

    let mut P: Vec<Vec<Vec<Vec<sunrealtype>>>> = Vec::with_capacity(MX as usize);
    let mut pivot: Vec<Vec<Vec<sunindextype>>> = Vec::with_capacity(MX as usize);
    for _jx in 0..MX {
        let mut Pcol: Vec<Vec<Vec<sunrealtype>>> = Vec::with_capacity(MY as usize);
        let mut pivotcol: Vec<Vec<sunindextype>> = Vec::with_capacity(MY as usize);
        for _jy in 0..MY {
            Pcol.push(vec![vec![ZERO; ns]; ns]);
            pivotcol.push(vec![0 as sunindextype; ns]);
        }
        P.push(Pcol);
        pivot.push(pivotcol);
    }

    Some(UserData {
        P,
        pivot,
        acoef: vec![vec![ZERO; ns]; ns],
        bcoef: vec![ZERO; ns],
        rates: None,
        cox: vec![ZERO; ns],
        coy: vec![ZERO; ns],
        ax: ZERO,
        ay: ZERO,
        dx: ZERO,
        dy: ZERO,
        uround: ZERO,
        sqruround: ZERO,
        mx: 0,
        my: 0,
        ns: 0,
        np: 0,
    })
}

/*
 * Load problem constants in data
 */

fn InitUserData(data: &mut UserData) {
    data.mx = MX;
    data.my = MY;
    data.ns = NUM_SPECIES;
    data.np = NUM_SPECIES / 2;
    data.ax = AX;
    data.ay = AY;
    data.dx = data.ax / (MX - 1) as sunrealtype;
    data.dy = data.ay / (MY - 1) as sunrealtype;
    data.uround = SUN_UNIT_ROUNDOFF;
    data.sqruround = data.uround.sqrt();

    /* Set up the coefficients a and b plus others found in the equations */
    let np = data.np as usize;

    let dx2 = data.dx * data.dx;
    let dy2 = data.dy * data.dy;

    for i in 0..np {
        /*  Fill in the portion of acoef in the four quadrants, row by row */
        for j in 0..np {
            data.acoef[i][np + j] = -GG;
            data.acoef[i + np][j] = EE;
            data.acoef[i][j] = ZERO;
            data.acoef[i + np][np + j] = ZERO;
        }

        /* and then change the diagonal elements of acoef to -AA */
        data.acoef[i][i] = -AA;
        data.acoef[i + np][i + np] = -AA;

        data.bcoef[i] = BB;
        data.bcoef[i + np] = -BB;

        data.cox[i] = DPREY / dx2;
        data.cox[i + np] = DPRED / dx2;

        data.coy[i] = DPREY / dy2;
        data.coy[i + np] = DPRED / dy2;
    }
}

/*
 * Set initial conditions in cc
 */

fn SetInitialProfiles(cc: &N_Vector, sc: &N_Vector) {
    let mut ctemp = [ZERO; NUM_SPECIES as usize];
    let mut stemp = [ZERO; NUM_SPECIES as usize];

    let mut ccdata = N_VGetArrayPointer(cc).expect("N_VGetArrayPointer");
    let mut scdata = N_VGetArrayPointer(sc).expect("N_VGetArrayPointer");

    /* Initialize arrays ctemp and stemp used in the loading process */
    for i in 0..(NUM_SPECIES / 2) as usize {
        ctemp[i] = PREYIN;
        stemp[i] = ONE;
    }
    for i in (NUM_SPECIES / 2) as usize..NUM_SPECIES as usize {
        ctemp[i] = PREDIN;
        stemp[i] = 0.00001;
    }

    /* Load initial profiles into cc and sc vector from ctemp and stemp. */
    for jy in 0..MY {
        for jx in 0..MX {
            let cloc = IJ_Vptr(jx, jy);
            let sloc = IJ_Vptr(jx, jy);
            for i in 0..NUM_SPECIES as usize {
                ccdata[cloc + i] = ctemp[i];
                scdata[sloc + i] = stemp[i];
            }
        }
    }
}

/*
 * Print first lines of output (problem description)
 */

fn PrintHeader(
    globalstrategy: i32,
    maxl: i32,
    maxlrst: i32,
    fnormtol: sunrealtype,
    scsteptol: sunrealtype,
    linsolver: i32,
) {
    print!("\nPredator-prey test problem --  KINSol (serial version)\n\n");
    print!("Mesh dimensions = {} X {}\n", MX, MY);
    print!("Number of species = {}\n", NUM_SPECIES);
    print!("Total system size = {}\n\n", NEQ);
    print!(
        "Flag globalstrategy = {} (0 = None, 1 = Linesearch)\n",
        globalstrategy
    );

    match linsolver {
        USE_SPGMR => {
            print!(
                "Linear solver is SPGMR with maxl = {}, maxlrst = {}\n",
                maxl, maxlrst
            );
        }

        USE_SPBCGS => {
            print!("Linear solver is SPBCGS with maxl = {}\n", maxl);
        }

        USE_SPTFQMR => {
            print!("Linear solver is SPTFQMR with maxl = {}\n", maxl);
        }

        USE_SPFGMR => {
            print!(
                "Linear solver is SPFGMR with maxl = {}, maxlrst = {}\n",
                maxl, maxlrst
            );
        }

        _ => {}
    }

    print!("Preconditioning uses interaction-only block-diagonal matrix\n");
    print!("Positivity constraints imposed on all components \n");
    print!(
        "Tolerance parameters:  fnormtol = {}   scsteptol = {}\n",
        fmt_g(fnormtol, 6),
        fmt_g(scsteptol, 6)
    );

    print!("\nInitial profile of concentration\n");
    print!(
        "At all mesh points:  {} {} {}   {} {} {}\n",
        fmt_g(PREYIN, 6),
        fmt_g(PREYIN, 6),
        fmt_g(PREYIN, 6),
        fmt_g(PREDIN, 6),
        fmt_g(PREDIN, 6),
        fmt_g(PREDIN, 6)
    );
}

/*
 * Print sampled values of current cc
 */

fn PrintOutput(cc: &N_Vector) {
    let ccdata = N_VGetArrayPointer(cc).expect("N_VGetArrayPointer");

    let jy = 0;
    let jx = 0;
    let ct = IJ_Vptr(jx, jy);
    print!("\nAt bottom left:");

    /* Print out lines with up to 6 values per line */
    for is in 0..NUM_SPECIES as usize {
        if (is % 6) * 6 == is {
            print!("\n");
        }
        print!(" {}", fmt_g(ccdata[ct + is], 6));
    }

    let jy = MY - 1;
    let jx = MX - 1;
    let ct = IJ_Vptr(jx, jy);
    print!("\n\nAt top right:");

    /* Print out lines with up to 6 values per line */
    for is in 0..NUM_SPECIES as usize {
        if (is % 6) * 6 == is {
            print!("\n");
        }
        print!(" {}", fmt_g(ccdata[ct + is], 6));
    }
    print!("\n\n");
}

/*
 * Print final statistics contained in iopt
 */

fn PrintFinalStats(kmem: &KINMem, linsolver: i32) {
    let mut nni: i64 = 0;
    let mut nfe: i64 = 0;
    let mut nli: i64 = 0;
    let mut npe: i64 = 0;
    let mut nps: i64 = 0;
    let mut ncfl: i64 = 0;
    let mut nfeSG: i64 = 0;
    let mut flag: i32;

    flag = KINGetNumNonlinSolvIters(kmem, &mut nni);
    check_flag(flag, "KINGetNumNonlinSolvIters", 1);
    flag = KINGetNumFuncEvals(kmem, &mut nfe);
    check_flag(flag, "KINGetNumFuncEvals", 1);

    flag = KINGetNumLinIters(kmem, &mut nli);
    check_flag(flag, "KINGetNumLinIters", 1);
    flag = KINGetNumPrecEvals(kmem, &mut npe);
    check_flag(flag, "KINGetNumPrecEvals", 1);
    flag = KINGetNumPrecSolves(kmem, &mut nps);
    check_flag(flag, "KINGetNumPrecSolves", 1);
    flag = KINGetNumLinConvFails(kmem, &mut ncfl);
    check_flag(flag, "KINGetNumLinConvFails", 1);
    flag = KINGetNumLinFuncEvals(kmem, &mut nfeSG);
    check_flag(flag, "KINGetNumLinFuncEvals", 1);

    print!("Final Statistics.. \n");
    print!("nni    = {:5}    nli   = {:5}\n", nni, nli);
    print!("nfe    = {:5}    nfeSG = {:5}\n", nfe, nfeSG);
    print!(
        "nps    = {:5}    npe   = {:5}     ncfl  = {:5}\n",
        nps, npe, ncfl
    );

    if linsolver < 3 {
        print!("\n=========================================================\n\n");
    }
}

/*
 * Check function return value...
 *    opt == 0 means SUNDIALS function allocates memory so check if
 *             returned NULL pointer
 *    opt == 1 means SUNDIALS function returns a flag so check if
 *             flag >= 0
 *    opt == 2 means function allocates memory so check if returned
 *             NULL pointer
 */

fn check_flag(flag: i32, funcname: &str, opt: i32) -> i32 {
    /* Check if flag < 0 */
    if opt == 1 && flag < 0 {
        eprint!(
            "\nSUNDIALS_ERROR: {}() failed with flag = {}\n\n",
            funcname, flag
        );
        return 1;
    }

    0
}

fn check_flag_ptr<T>(flagvalue: &Option<T>, funcname: &str, opt: i32) -> i32 {
    /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
    if opt == 0 && flagvalue.is_none() {
        eprint!(
            "\nSUNDIALS_ERROR: {}() failed - returned NULL pointer\n\n",
            funcname
        );
        return 1;
    }
    /* Check if function returned NULL pointer - no memory allocated */
    else if opt == 2 && flagvalue.is_none() {
        eprint!(
            "\nMEMORY_ERROR: {}() failed - returned NULL pointer\n\n",
            funcname
        );
        return 1;
    }

    0
}
