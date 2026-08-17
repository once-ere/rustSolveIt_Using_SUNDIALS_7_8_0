#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

/* -----------------------------------------------------------------
 * Ported from: examples/ida/serial/idaFoodWeb_bnd.c
 * Programmer(s): Allan Taylor, Alan Hindmarsh and
 *                Radu Serban @ LLNL
 * -----------------------------------------------------------------
 * Example program for IDA: Food web problem.
 *
 * This example program (serial version) uses the banded linear
 * solver, and IDACalcIC for initial condition calculation.
 *
 * The mathematical problem solved in this example is a DAE system
 * that arises from a system of partial differential equations after
 * spatial discretization. The PDE system is a food web population
 * model, with predator-prey interaction and diffusion on the unit
 * square in two dimensions.
 *
 * The number of species is ns = 2 * np, with the first np being
 * prey and the last np being predators. In this program, np = 1,
 * ns = 2. The boundary conditions are homogeneous Neumann:
 * normal derivative = 0.
 *
 * The PDEs are discretized by central differencing on a MX by MY
 * mesh.
 *
 * The DAE system is solved by IDA using the banded linear solver.
 * Output is printed at t = 0, .001, .01, .1, .4, .7, 1.
 * -----------------------------------------------------------------*/

use std::any::Any;

use ida_rs::prelude::*;

/* Problem Constants. */

const NPREY: usize = 1; /* No. of prey (= no. of predators). */
const NUM_SPECIES: usize = 2 * NPREY;

const PI: sunrealtype = 3.1415926535898;
const FOURPI: sunrealtype = 4.0 * PI;

const MX: usize = 20; /* MX = number of x mesh points      */
const MY: usize = 20; /* MY = number of y mesh points      */
const NSMX: usize = NUM_SPECIES * MX;
const NEQ: usize = NUM_SPECIES * MX * MY;
const AA: sunrealtype = 1.0; /* Coefficient in above eqns. for a  */
const EE: sunrealtype = 10000.; /* Coefficient in above eqns. for a  */
const GG: sunrealtype = 0.5e-6; /* Coefficient in above eqns. for a  */
const BB: sunrealtype = 1.0; /* Coefficient in above eqns. for b  */
const DPREY: sunrealtype = 1.0; /* Coefficient in above eqns. for d  */
const DPRED: sunrealtype = 0.05; /* Coefficient in above eqns. for d  */
const ALPHA: sunrealtype = 50.; /* Coefficient alpha in above eqns.  */
const BETA: sunrealtype = 1000.; /* Coefficient beta in above eqns.   */
const AX: sunrealtype = 1.0; /* Total range of x variable         */
const AY: sunrealtype = 1.0; /* Total range of y variable         */
const RTOL: sunrealtype = 1.0e-5; /* Relative tolerance                */
const ATOL: sunrealtype = 1.0e-5; /* Absolute tolerance                */
const NOUT: i32 = 6; /* Number of output times            */
const TMULT: sunrealtype = 10.0; /* Multiplier for tout values        */
const TADD: sunrealtype = 0.3; /* Increment for tout values         */
const ZERO: sunrealtype = 0.;
const ONE: sunrealtype = 1.0;

/*
 * User-defined vector and accessor macro: IJ_Vptr.
 * IJ_Vptr is defined in order to express the underlying 3-D structure of
 * the dependent variable vector from its underlying 1-D storage (an N_Vector).
 * IJ_Vptr(vv,i,j) returns the index into vv corresponding to
 * species index is = 0, x-index ix = i, and y-index jy = j.
 */

fn IJ_Vptr(i: usize, j: usize) -> usize {
    i * NUM_SPECIES + j * NSMX
}

/* Type: UserData.  Contains problem constants, etc. */

#[allow(dead_code)]
struct UserData {
    Neq: sunindextype,
    ns: sunindextype,
    np: usize,
    mx: sunindextype,
    my: sunindextype,
    dx: sunrealtype,
    dy: sunrealtype,
    acoef: [sunrealtype; NUM_SPECIES * NUM_SPECIES],
    cox: [sunrealtype; NUM_SPECIES],
    coy: [sunrealtype; NUM_SPECIES],
    bcoef: [sunrealtype; NUM_SPECIES],
    rates: Option<N_Vector>,
}

/*
 *--------------------------------------------------------------------
 * MAIN PROGRAM
 *--------------------------------------------------------------------
 */

fn main() {
    let mut retval: i32;
    let mu: sunindextype;
    let ml: sunindextype;
    let rtol: sunrealtype;
    let atol: sunrealtype;
    let t0: sunrealtype;
    let mut tout: sunrealtype;
    let mut tret: sunrealtype = 0.0;

    /* Create the SUNDIALS context object for this simulation */
    let mut ctx: Option<SUNContext> = None;
    retval = SUNContext_Create(SUN_COMM_NULL, &mut ctx);
    if check_retval(&retval, "SUNContext_Create") != 0 {
        std::process::exit(1);
    }
    let ctx = ctx.expect("SUNContext_Create");

    /* Allocate and initialize user data block webdata. */

    let mut webdata = UserData {
        Neq: 0,
        ns: 0,
        np: 0,
        mx: 0,
        my: 0,
        dx: ZERO,
        dy: ZERO,
        acoef: [ZERO; NUM_SPECIES * NUM_SPECIES],
        cox: [ZERO; NUM_SPECIES],
        coy: [ZERO; NUM_SPECIES],
        bcoef: [ZERO; NUM_SPECIES],
        rates: None,
    };
    webdata.rates = N_VNew_Serial(NEQ as sunindextype, &ctx);

    InitUserData(&mut webdata);

    /* Allocate N-vectors and initialize cc, cp, and id. */

    let cc = N_VNew_Serial(NEQ as sunindextype, &ctx);
    if check_ptr(&cc, "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    let cc = cc.expect("N_VNew_Serial");

    let cp = N_VClone(&cc);
    if check_ptr(&cp, "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    let cp = cp.expect("N_VClone");

    let id = N_VClone(&cc);
    if check_ptr(&id, "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    let id = id.expect("N_VClone");

    SetInitialProfiles(&cc, &cp, &id, &webdata);

    /* Set remaining inputs to IDAMalloc. */

    t0 = ZERO;
    rtol = RTOL;
    atol = ATOL;

    /* Call IDACreate and IDAMalloc to initialize IDA. */

    let mem = IDACreate(&ctx);
    if check_ptr(&mem, "IDACreate", 0) != 0 {
        std::process::exit(1);
    }
    let mem = mem.expect("IDACreate");

    retval = IDASetUserData(&mem, Some(Box::new(webdata)));
    if check_retval(&retval, "IDASetUserData") != 0 {
        std::process::exit(1);
    }

    retval = IDASetId(&mem, Some(&id));
    if check_retval(&retval, "IDASetId") != 0 {
        std::process::exit(1);
    }

    retval = IDAInit(&mem, resweb, t0, &cc, &cp);
    if check_retval(&retval, "IDAInit") != 0 {
        std::process::exit(1);
    }

    retval = IDASStolerances(&mem, rtol, atol);
    if check_retval(&retval, "IDASStolerances") != 0 {
        std::process::exit(1);
    }

    /* Create banded SUNMatrix for use in linear solves */
    mu = NSMX as sunindextype;
    ml = NSMX as sunindextype;
    let A = SUNBandMatrix(NEQ as sunindextype, mu, ml, &ctx);
    if check_ptr(&A, "SUNBandMatrix", 0) != 0 {
        std::process::exit(1);
    }
    let A = A.expect("SUNBandMatrix");

    /* Create banded SUNLinearSolver object */
    let LS = SUNLinSol_Band(&cc, &A, &ctx);
    if check_ptr(&LS, "SUNLinSol_Band", 0) != 0 {
        std::process::exit(1);
    }
    let LS = LS.expect("SUNLinSol_Band");

    /* Attach the matrix and linear solver */
    retval = IDASetLinearSolver(&mem, &LS, Some(&A));
    if check_retval(&retval, "IDASetLinearSolver") != 0 {
        std::process::exit(1);
    }

    /* Call IDACalcIC (with default options) to correct the initial values. */

    tout = 0.001;
    retval = IDACalcIC(&mem, IDA_YA_YDP_INIT, tout);
    if check_retval(&retval, "IDACalcIC") != 0 {
        std::process::exit(1);
    }

    /* Print heading, basic parameters, and initial values. */

    PrintHeader(mu, ml, rtol, atol);
    PrintOutput(&mem, &cc, ZERO);

    /* Loop over iout, call IDASolve (normal mode), print selected output. */

    let mut iout = 1;
    while iout <= NOUT {
        retval = IDASolve(&mem, tout, &mut tret, &cc, &cp, IDA_NORMAL);
        if check_retval(&retval, "IDASolve") != 0 {
            std::process::exit(retval);
        }

        PrintOutput(&mem, &cc, tret);

        if iout < 3 {
            tout *= TMULT;
        } else {
            tout += TADD;
        }
        iout += 1;
    }

    /* Print final statistics and free memory. */

    PrintFinalStats(&mem);

    /* Free memory */

    IDAFree(&mut Some(mem));
    let _ = SUNLinSolFree(Some(LS));
    SUNMatDestroy(A);

    N_VDestroy(cc);
    N_VDestroy(cp);
    N_VDestroy(id);

    /* webdata (with acoef and rates) is owned by the IDA memory record */

    let _ = SUNContext_Free(&mut Some(ctx));
}

/*
 *--------------------------------------------------------------------
 * FUNCTIONS CALLED BY IDA
 *--------------------------------------------------------------------
 */

/*
 * resweb: System residual function for predator-prey system.
 * This routine calls Fweb to get all the right-hand sides of the
 * equations, then loads the residual vector accordingly,
 * using cp in the case of prey species.
 */

fn resweb(
    tt: sunrealtype,
    cc: &N_Vector,
    cp: &N_Vector,
    res: &N_Vector,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let webdata = user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<UserData>())
        .expect("user_data is UserData");

    let np = webdata.np;

    /* Call Fweb to set res to vector of right-hand sides. */
    Fweb(tt, cc, res, webdata);

    /* Loop over all grid points, setting residual values appropriately
    for differential or algebraic components.                        */

    {
        let cpv = N_VGetArrayPointer(cp).expect("N_VGetArrayPointer");
        let mut resv = N_VGetArrayPointer(res).expect("N_VGetArrayPointer");

        for jy in 0..MY {
            let yloc = NSMX * jy;
            for jx in 0..MX {
                let loc = yloc + NUM_SPECIES * jx;
                for is in 0..NUM_SPECIES {
                    if is < np {
                        resv[loc + is] = cpv[loc + is] - resv[loc + is];
                    } else {
                        resv[loc + is] = -resv[loc + is];
                    }
                }
            }
        }
    }

    0
}

/*
 *--------------------------------------------------------------------
 * PRIVATE FUNCTIONS
 *--------------------------------------------------------------------
 */

/*
 * InitUserData: Load problem constants in webdata (of type UserData).
 */

fn InitUserData(webdata: &mut UserData) {
    webdata.mx = MX as sunindextype;
    webdata.my = MY as sunindextype;
    webdata.ns = NUM_SPECIES as sunindextype;
    webdata.np = NPREY;
    webdata.dx = AX / ((MX - 1) as sunrealtype);
    webdata.dy = AY / ((MY - 1) as sunrealtype);
    webdata.Neq = NEQ as sunindextype;

    /* Set up the coefficients a and b, and others found in the equations. */
    let np = webdata.np;
    let dx2 = webdata.dx * webdata.dx;
    let dy2 = webdata.dy * webdata.dy;

    for i in 0..np {
        /* Fill in the portion of acoef in the four quadrants, row by row.
        acoef[i][j] is stored at acoef[i * NUM_SPECIES + j]. */
        for j in 0..np {
            webdata.acoef[i * NUM_SPECIES + (np + j)] = -GG;
            webdata.acoef[(i + np) * NUM_SPECIES + j] = EE;
            webdata.acoef[i * NUM_SPECIES + j] = ZERO;
            webdata.acoef[(i + np) * NUM_SPECIES + (np + j)] = ZERO;
        }

        /* Reset the diagonal elements of acoef to -AA. */
        webdata.acoef[i * NUM_SPECIES + i] = -AA;
        webdata.acoef[(i + np) * NUM_SPECIES + (i + np)] = -AA;

        /* Set coefficients for b and diffusion terms. */
        webdata.bcoef[i] = BB;
        webdata.bcoef[i + np] = -BB;
        webdata.cox[i] = DPREY / dx2;
        webdata.cox[i + np] = DPRED / dx2;
        webdata.coy[i] = DPREY / dy2;
        webdata.coy[i + np] = DPRED / dy2;
    }
}

/*
 * SetInitialProfiles: Set initial conditions in cc, cp, and id.
 * A polynomial profile is used for the prey cc values, and a constant
 * (1.0e5) is loaded as the initial guess for the predator cc values.
 * The id values are set to 1 for the prey and 0 for the predators.
 * The prey cp values are set according to the given system, and
 * the predator cp values are set to zero.
 */

fn SetInitialProfiles(cc: &N_Vector, cp: &N_Vector, id: &N_Vector, webdata: &UserData) {
    let np = webdata.np;

    /* Loop over grid, load cc values and id values. */
    {
        let mut ccv = N_VGetArrayPointer(cc).expect("N_VGetArrayPointer");
        let mut idv = N_VGetArrayPointer(id).expect("N_VGetArrayPointer");

        for jy in 0..MY {
            let yy = (jy as sunrealtype) * webdata.dy;
            let yloc = NSMX * jy;
            for jx in 0..MX {
                let xx = (jx as sunrealtype) * webdata.dx;
                let mut xyfactor = 16.0 * xx * (ONE - xx) * yy * (ONE - yy);
                xyfactor *= xyfactor;
                let loc = yloc + NUM_SPECIES * jx;

                for is in 0..NUM_SPECIES {
                    if is < np {
                        ccv[loc + is] = 10.0 + ((is + 1) as sunrealtype) * xyfactor;
                        idv[loc + is] = ONE;
                    } else {
                        ccv[loc + is] = 1.0e5;
                        idv[loc + is] = ZERO;
                    }
                }
            }
        }
    }

    /* Set c' for the prey by calling the function Fweb. */
    Fweb(ZERO, cc, cp, webdata);

    /* Set c' for predators to 0. */
    {
        let mut cpv = N_VGetArrayPointer(cp).expect("N_VGetArrayPointer");

        for jy in 0..MY {
            let yloc = NSMX * jy;
            for jx in 0..MX {
                let loc = yloc + NUM_SPECIES * jx;
                for is in np..NUM_SPECIES {
                    cpv[loc + is] = ZERO;
                }
            }
        }
    }
}

/*
 * Print first lines of output (problem description)
 */

fn PrintHeader(mu: sunindextype, ml: sunindextype, rtol: sunrealtype, atol: sunrealtype) {
    print!("\nidaFoodWeb_bnd: Predator-prey DAE serial example problem for IDA \n\n");
    print!("Number of species ns: {}", NUM_SPECIES);
    print!("     Mesh dimensions: {} x {}", MX, MY);
    print!("     System size: {}\n", NEQ);
    print!(
        "Tolerance parameters:  rtol = {}   atol = {}\n",
        fmt_g(rtol, 6),
        fmt_g(atol, 6)
    );
    print!(
        "Linear solver: BAND,  Band parameters mu = {}, ml = {}\n",
        mu, ml
    );
    print!("CalcIC called to correct initial predator concentrations.\n\n");
    print!("-----------------------------------------------------------\n");
    print!("  t        bottom-left  top-right");
    print!("    | nst  k      h\n");
    print!("-----------------------------------------------------------\n\n");
}

/*
 * PrintOutput: Print output values at output time t = tt.
 * Selected run statistics are printed.  Then values of the concentrations
 * are printed for the bottom left and top right grid points only.
 */

fn PrintOutput(mem: &IDAMem, c: &N_Vector, t: sunrealtype) {
    let mut retval: i32;
    let mut kused: i32 = 0;
    let mut nst: i64 = 0;
    let mut hused: sunrealtype = 0.0;

    retval = IDAGetLastOrder(mem, &mut kused);
    check_retval(&retval, "IDAGetLastOrder");
    retval = IDAGetNumSteps(mem, &mut nst);
    check_retval(&retval, "IDAGetNumSteps");
    retval = IDAGetLastStep(mem, &mut hused);
    check_retval(&retval, "IDAGetLastStep");

    let mut c_bl = [ZERO; NUM_SPECIES];
    let mut c_tr = [ZERO; NUM_SPECIES];
    {
        let cv = N_VGetArrayPointer(c).expect("N_VGetArrayPointer");
        let bl = IJ_Vptr(0, 0);
        let tr = IJ_Vptr(MX - 1, MY - 1);
        for i in 0..NUM_SPECIES {
            c_bl[i] = cv[bl + i];
            c_tr[i] = cv[tr + i];
        }
    }

    print!(
        "{} {} {}   | {:>3}  {:>1} {}\n",
        fmt_ew(t, 8, 2),
        fmt_ew(c_bl[0], 12, 4),
        fmt_ew(c_tr[0], 12, 4),
        nst,
        kused,
        fmt_ew(hused, 12, 4)
    );
    for i in 1..NUM_SPECIES {
        print!(
            "         {} {}   |\n",
            fmt_ew(c_bl[i], 12, 4),
            fmt_ew(c_tr[i], 12, 4)
        );
    }

    print!("\n");
}

/*
 * PrintFinalStats: Print final run data contained in iopt.
 */

fn PrintFinalStats(mem: &IDAMem) {
    let mut retval: i32;
    let mut nst: i64 = 0;
    let mut nre: i64 = 0;
    let mut nreLS: i64 = 0;
    let mut nni: i64 = 0;
    let mut nnf: i64 = 0;
    let mut nje: i64 = 0;
    let mut netf: i64 = 0;
    let mut ncfn: i64 = 0;

    retval = IDAGetNumSteps(mem, &mut nst);
    check_retval(&retval, "IDAGetNumSteps");
    retval = IDAGetNumNonlinSolvIters(mem, &mut nni);
    check_retval(&retval, "IDAGetNumNonlinSolvIters");
    retval = IDAGetNumResEvals(mem, &mut nre);
    check_retval(&retval, "IDAGetNumResEvals");
    retval = IDAGetNumErrTestFails(mem, &mut netf);
    check_retval(&retval, "IDAGetNumErrTestFails");
    retval = IDAGetNumNonlinSolvConvFails(mem, &mut nnf);
    check_retval(&retval, "IDAGetNumNonlinSolvConvFails");
    retval = IDAGetNumStepSolveFails(mem, &mut ncfn);
    check_retval(&retval, "IDAGetNumStepSolveFails");
    retval = IDAGetNumJacEvals(mem, &mut nje);
    check_retval(&retval, "IDAGetNumJacEvals");
    retval = IDAGetNumLinResEvals(mem, &mut nreLS);
    check_retval(&retval, "IDAGetNumLinResEvals");

    print!("-----------------------------------------------------------\n");
    print!("Final run statistics: \n\n");
    print!("Number of steps                    = {}\n", nst);
    print!("Number of residual evaluations     = {}\n", nre + nreLS);
    print!("Number of Jacobian evaluations     = {}\n", nje);
    print!("Number of nonlinear iterations     = {}\n", nni);
    print!("Number of error test failures      = {}\n", netf);
    print!("Number of nonlinear conv. failures = {}\n", nnf);
    print!("Number of step solver failures     = {}\n", ncfn);
}

/*
 * Fweb: Rate function for the food-web problem.
 * This routine computes the right-hand sides of the system equations,
 * consisting of the diffusion term and interaction term.
 * The interaction term is computed by the function WebRates.
 */

fn Fweb(_tcalc: sunrealtype, cc: &N_Vector, crate_: &N_Vector, webdata: &UserData) {
    let rates = webdata.rates.as_ref().expect("rates").clone();

    let ccv = N_VGetArrayPointer(cc).expect("N_VGetArrayPointer");
    let mut ratesv = N_VGetArrayPointer(&rates).expect("N_VGetArrayPointer");
    let mut cratev = N_VGetArrayPointer(crate_).expect("N_VGetArrayPointer");

    /* Loop over grid points, evaluate interaction vector (length ns),
    form diffusion difference terms, and load crate.                    */

    for jy in 0..MY {
        let yy = webdata.dy * (jy as sunrealtype);
        let idyu: isize = if jy != MY - 1 {
            NSMX as isize
        } else {
            -(NSMX as isize)
        };
        let idyl: isize = if jy != 0 {
            NSMX as isize
        } else {
            -(NSMX as isize)
        };

        for jx in 0..MX {
            let xx = webdata.dx * (jx as sunrealtype);
            let idxu: isize = if jx != MX - 1 {
                NUM_SPECIES as isize
            } else {
                -(NUM_SPECIES as isize)
            };
            let idxl: isize = if jx != 0 {
                NUM_SPECIES as isize
            } else {
                -(NUM_SPECIES as isize)
            };
            let cxy = IJ_Vptr(jx, jy);
            let ratesxy = IJ_Vptr(jx, jy);
            let cratexy = IJ_Vptr(jx, jy);

            /* Get interaction vector at this grid point. */
            WebRates(
                xx,
                yy,
                &ccv[cxy..cxy + NUM_SPECIES],
                &mut ratesv[ratesxy..ratesxy + NUM_SPECIES],
                &webdata.acoef,
                &webdata.bcoef,
            );

            /* Loop over species, do differencing, load crate segment. */
            for is in 0..NUM_SPECIES {
                let base = cxy as isize + is as isize;

                /* Differencing in y. */
                let dcyli = ccv[base as usize] - ccv[(base - idyl) as usize];
                let dcyui = ccv[(base + idyu) as usize] - ccv[base as usize];

                /* Differencing in x. */
                let dcxli = ccv[base as usize] - ccv[(base - idxl) as usize];
                let dcxui = ccv[(base + idxu) as usize] - ccv[base as usize];

                /* Compute the crate values at (xx,yy). */
                cratev[cratexy + is] = webdata.coy[is] * (dcyui - dcyli)
                    + webdata.cox[is] * (dcxui - dcxli)
                    + ratesv[ratesxy + is];
            } /* End is loop */
        } /* End of jx loop */
    } /* End of jy loop */
}

/*
 * WebRates: Evaluate reaction rates at a given spatial point.
 * At a given (x,y), evaluate the array of ns reaction terms R.
 */

fn WebRates(
    xx: sunrealtype,
    yy: sunrealtype,
    cxy: &[sunrealtype],
    ratesxy: &mut [sunrealtype],
    acoef: &[sunrealtype],
    bcoef: &[sunrealtype],
) {
    for is in 0..NUM_SPECIES {
        ratesxy[is] = dotprod(
            NUM_SPECIES as sunindextype,
            cxy,
            &acoef[is * NUM_SPECIES..is * NUM_SPECIES + NUM_SPECIES],
        );
    }

    let fac = ONE + ALPHA * xx * yy + BETA * (FOURPI * xx).sun_sin() * (FOURPI * yy).sun_sin();

    for is in 0..NUM_SPECIES {
        ratesxy[is] = cxy[is] * (bcoef[is] * fac + ratesxy[is]);
    }
}

/*
 * dotprod: dot product routine for sunrealtype arrays, for use by WebRates.
 */

fn dotprod(size: sunindextype, x1: &[sunrealtype], x2: &[sunrealtype]) -> sunrealtype {
    let mut temp = ZERO;
    for i in 0..size as usize {
        temp += x1[i] * x2[i];
    }
    temp
}

/*
 * Check function return value...
 *   opt == 0 means SUNDIALS function allocates memory so check if
 *            returned NULL pointer
 *   opt == 1 means SUNDIALS function returns an integer value so check if
 *            retval < 0
 *   opt == 2 means function allocates memory so check if returned
 *            NULL pointer
 */

fn check_retval(retval: &i32, funcname: &str) -> i32 {
    /* Check if retval < 0 */
    if *retval < 0 {
        eprint!(
            "\nSUNDIALS_ERROR: {}() failed with retval = {}\n\n",
            funcname, retval
        );
        return 1;
    }
    0
}

fn check_ptr<T>(returnvalue: &Option<T>, funcname: &str, opt: i32) -> i32 {
    if returnvalue.is_none() {
        if opt == 0 {
            /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
            eprint!(
                "\nSUNDIALS_ERROR: {}() failed - returned NULL pointer\n\n",
                funcname
            );
        } else {
            /* Check if function returned NULL pointer - no memory allocated */
            eprint!(
                "\nMEMORY_ERROR: {}() failed - returned NULL pointer\n\n",
                funcname
            );
        }
        return 1;
    }
    0
}
