#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

/* -----------------------------------------------------------------
 * Ported from: examples/ida/serial/idaHeat2D_klu.c
 * Programmer(s): Allan Taylor, Alan Hindmarsh and
 *                Radu Serban @ LLNL
 * -----------------------------------------------------------------
 * Example problem for IDA: 2D heat equation, serial, sparse.
 *
 * This example solves a discretized 2D heat equation problem.
 * This version uses the KLU solver and IDACalcIC.
 *
 * Solver note: SUNLinSol_KLU here is sundials_core::sunlinsol_klu,
 * backed by the independent pure-Rust sparse LU rather than
 * SuiteSparse KLU. See differences/ATTRIBUTION.md.
 *
 * The DAE system solved is a spatial discretization of the PDE
 *          du/dt = d^2u/dx^2 + d^2u/dy^2
 * on the unit square. The boundary condition is u = 0 on all edges.
 * Initial conditions are given by u = 16 x (1 - x) y (1 - y).
 * The PDE is treated with central differences on a uniform M x M
 * grid. The values of u at the interior points satisfy ODEs, and
 * equations u = 0 at the boundaries are appended, to form a DAE
 * system of size N = M^2. Here M = 10.
 *
 * The system is solved with IDA using the KLU sparse direct
 * linear solver and a user-supplied Jacobian. For purposes of
 * illustration,
 * IDACalcIC is called to compute correct values at the boundary,
 * given incorrect values as input initial guesses. The constraints
 * u >= 0 are posed for all components. Output is taken at
 * t = 0, .01, .02, .04, ..., 10.24. (Output at t = 0 is for
 * IDACalcIC cost statistics only.)
 * -----------------------------------------------------------------*/

use std::any::Any;

use ida_rs::prelude::*;

/* Problem Constants */

const NOUT: i32 = 11;
const MGRID: sunindextype = 10;
const NEQ: sunindextype = MGRID * MGRID;
/* total num of nonzero elements */
const TOTAL: sunindextype =
    4 * MGRID + 8 * (MGRID - 2) + (MGRID - 4) * (MGRID + 4 * (MGRID - 2));
const ZERO: sunrealtype = 0.0;
const ONE: sunrealtype = 1.0;
const TWO: sunrealtype = 2.0;
const BVAL: sunrealtype = 0.0;

/* Type: UserData */

struct UserData {
    mm: sunindextype,
    dx: sunrealtype,
    coeff: sunrealtype,
}

/*
 *--------------------------------------------------------------------
 * MAIN PROGRAM
 *--------------------------------------------------------------------
 */

fn main() {
    let mut retval: i32;
    let mut netf: i64 = 0;
    let mut ncfn: i64 = 0;
    let rtol: sunrealtype;
    let atol: sunrealtype;
    let t0: sunrealtype;
    let t1: sunrealtype;
    let mut tout: sunrealtype;
    let mut tret: sunrealtype = 0.0;

    /* Create the SUNDIALS context object for this simulation */
    let mut ctx: Option<SUNContext> = None;
    retval = SUNContext_Create(SUN_COMM_NULL, &mut ctx);
    if check_retval(&retval, "SUNContext_Create") != 0 {
        std::process::exit(1);
    }
    let ctx = ctx.expect("SUNContext_Create");

    /* Create vectors uu, up, res, constraints, id. */
    let uu = N_VNew_Serial(NEQ, &ctx);
    if check_ptr(&uu, "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    let uu = uu.expect("N_VNew_Serial");
    let up = N_VClone(&uu);
    if check_ptr(&up, "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    let up = up.expect("N_VClone");
    let res = N_VClone(&uu);
    if check_ptr(&res, "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    let res = res.expect("N_VClone");
    let constraints = N_VClone(&uu);
    if check_ptr(&constraints, "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    let constraints = constraints.expect("N_VClone");
    let id = N_VClone(&uu);
    if check_ptr(&id, "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    let id = id.expect("N_VClone");

    /* Create and load problem data block. */
    let dx = ONE / ((MGRID - 1) as sunrealtype);
    let mut data: Option<Box<dyn Any>> = Some(Box::new(UserData {
        mm: MGRID,
        dx,
        coeff: ONE / (dx * dx),
    }));
    if check_ptr(&data, "malloc", 2) != 0 {
        std::process::exit(1);
    }

    /* Initialize uu, up, id. */
    SetInitialProfile(&mut data, &uu, &up, &id, &res);

    /* Set constraints to all 1's for nonnegative solution values. */
    N_VConst(ONE, &constraints);

    /* Set remaining input parameters. */
    t0 = ZERO;
    t1 = 0.01;
    rtol = ZERO;
    atol = 1.0e-8;

    /* Call IDACreate and IDAMalloc to initialize solution */
    let mem = IDACreate(&ctx);
    if check_ptr(&mem, "IDACreate", 0) != 0 {
        std::process::exit(1);
    }
    let mem = mem.expect("IDACreate");

    retval = IDASetUserData(&mem, data.take());
    if check_retval(&retval, "IDASetUserData") != 0 {
        std::process::exit(1);
    }

    /* Set which components are algebraic or differential */
    retval = IDASetId(&mem, Some(&id));
    if check_retval(&retval, "IDASetId") != 0 {
        std::process::exit(1);
    }

    retval = IDASetConstraints(&mem, Some(&constraints));
    if check_retval(&retval, "IDASetConstraints") != 0 {
        std::process::exit(1);
    }
    N_VDestroy(constraints);

    retval = IDAInit(&mem, heatres, t0, &uu, &up);
    if check_retval(&retval, "IDAInit") != 0 {
        std::process::exit(1);
    }

    retval = IDASStolerances(&mem, rtol, atol);
    if check_retval(&retval, "IDASStolerances") != 0 {
        std::process::exit(1);
    }

    /* Create banded SUNMatrix for use in linear solves */
    let nnz: sunindextype = NEQ * NEQ;
    let A = SUNSparseMatrix(NEQ, NEQ, nnz, SUN_CSC_MAT, &ctx);
    if check_ptr(&A, "SUNSparseMtarix", 0) != 0 {
        std::process::exit(1);
    }
    let A = A.expect("SUNSparseMatrix");

    /* Create KLU SUNLinearSolver object */
    let LS = SUNLinSol_KLU(&uu, &A, &ctx);
    if check_ptr(&LS, "SUNLinSol_KLU", 0) != 0 {
        std::process::exit(1);
    }
    let LS = LS.expect("SUNLinSol_KLU");

    /* Attach the matrix and linear solver */
    retval = IDASetLinearSolver(&mem, &LS, Some(&A));
    if check_retval(&retval, "IDASetLinearSolver") != 0 {
        std::process::exit(1);
    }

    /* Set the user-supplied Jacobian routine */
    if MGRID >= 4 {
        retval = IDASetJacFn(&mem, Some(jacHeat));
    } else if MGRID == 3 {
        retval = IDASetJacFn(&mem, Some(jacHeat3));
    } else {
        /* MGRID<=2 is pure boundary points, nothing to solve */
        print!("MGRID size is too small to run.\n");
        std::process::exit(1);
    }
    if check_retval(&retval, "IDASetJacFn") != 0 {
        std::process::exit(1);
    }

    /* Call IDACalcIC to correct the initial values. */

    retval = IDACalcIC(&mem, IDA_YA_YDP_INIT, t1);
    if check_retval(&retval, "IDACalcIC") != 0 {
        std::process::exit(1);
    }

    /* Print output heading. */
    PrintHeader(rtol, atol);

    PrintOutput(&mem, t0, &uu);

    /* Loop over output times, call IDASolve, and print results. */

    tout = t1;
    let mut iout = 1;
    while iout <= NOUT {
        retval = IDASolve(&mem, tout, &mut tret, &uu, &up, IDA_NORMAL);
        if check_retval(&retval, "IDASolve") != 0 {
            std::process::exit(1);
        }

        PrintOutput(&mem, tret, &uu);

        iout += 1;
        tout *= TWO;
    }

    /* Print remaining counters and free memory. */
    retval = IDAGetNumErrTestFails(&mem, &mut netf);
    check_retval(&retval, "IDAGetNumErrTestFails");
    retval = IDAGetNumNonlinSolvConvFails(&mem, &mut ncfn);
    check_retval(&retval, "IDAGetNumNonlinSolvConvFails");
    print!("\n netf = {},   ncfn = {} \n", netf, ncfn);

    IDAFree(&mut Some(mem));
    let _ = SUNLinSolFree(Some(LS));
    SUNMatDestroy(A);
    N_VDestroy(uu);
    N_VDestroy(up);
    N_VDestroy(id);
    N_VDestroy(res);
    /* free(data) -- the UserData box is owned by the IDA memory */

    let _ = SUNContext_Free(&mut Some(ctx));
}

/*
 *--------------------------------------------------------------------
 * FUNCTIONS CALLED BY IDA
 *--------------------------------------------------------------------
 */

/*
 * heatres: heat equation system residual function
 * This uses 5-point central differencing on the interior points, and
 * includes algebraic equations for the boundary values.
 * So for each interior point, the residual component has the form
 *    res_i = u'_i - (central difference)_i
 * while for each boundary point, it is res_i = u_i.
 */

fn heatres(
    _tres: sunrealtype,
    uu: &N_Vector,
    up: &N_Vector,
    resval: &N_Vector,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let (mm, coeff) = {
        let data = user_data
            .as_mut()
            .and_then(|b| b.downcast_mut::<UserData>())
            .expect("user_data is UserData");
        (data.mm, data.coeff)
    };

    /* Initialize resval to uu, to take care of boundary equations.
    (The KLU variant of this example scales by ZERO where the banded one
    scales by ONE. Translated as written.) */
    N_VScale(ZERO, uu, resval);

    /* Loop over interior points; set res = up - (central difference). */
    {
        let uv = N_VGetArrayPointer(uu).expect("N_VGetArrayPointer");
        let upv = N_VGetArrayPointer(up).expect("N_VGetArrayPointer");
        let mut resv = N_VGetArrayPointer(resval).expect("N_VGetArrayPointer");

        for j in 1..(mm - 1) {
            let offset = mm * j;
            for i in 1..(mm - 1) {
                let loc = (offset + i) as usize;
                resv[loc] = upv[loc]
                    - coeff
                        * (uv[loc - 1]
                            + uv[loc + 1]
                            + uv[loc - mm as usize]
                            + uv[loc + mm as usize]
                            - 4.0 * uv[loc]);
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
 * Jacobian matrix setup for MGRID = 3. Translation of `jacHeat3`.
 */

fn jacHeat3(
    _tt: sunrealtype,
    cj: sunrealtype,
    _yy: &N_Vector,
    _yp: &N_Vector,
    _resvec: &N_Vector,
    JJ: &SUNMatrix,
    _user_data: &mut Option<Box<dyn Any>>,
    _tempv1: &N_Vector,
    _tempv2: &N_Vector,
    _tempv3: &N_Vector,
) -> i32 {
    let dx = ONE / (MGRID as sunrealtype - ONE);
    let beta = 4.0 / (dx * dx) + cj;

    SUNMatZero(JJ);

    /* One borrow for all three arrays: taking them separately would be a
    second mutable borrow of the same matrix content. */
    let mut m = SUNSparseMatrix_Content(JJ);
    let m = &mut *m;
    let (data, rowvals, colptrs) = (&mut m.data, &mut m.indexvals, &mut m.indexptrs);

    /* set up number of elements in each column */
    colptrs[0] = 0;
    colptrs[1] = 1;
    colptrs[2] = 3;
    colptrs[3] = 4;
    colptrs[4] = 6;
    colptrs[5] = 7;
    colptrs[6] = 9;
    colptrs[7] = 10;
    colptrs[8] = 12;
    colptrs[9] = 13;

    /* set up data and row values stored */
    data[0] = ONE;
    rowvals[0] = 0;
    data[1] = ONE;
    rowvals[1] = 1;
    data[2] = -ONE / (dx * dx);
    rowvals[2] = 4;
    data[3] = ONE;
    rowvals[3] = 2;
    data[4] = ONE;
    rowvals[4] = 3;
    data[5] = -ONE / (dx * dx);
    rowvals[5] = 4;
    data[6] = beta;
    rowvals[6] = 4;
    data[7] = -ONE / (dx * dx);
    rowvals[7] = 4;
    data[8] = ONE;
    rowvals[8] = 5;
    data[9] = ONE;
    rowvals[9] = 6;
    data[10] = -ONE / (dx * dx);
    rowvals[10] = 4;
    data[11] = ONE;
    rowvals[11] = 7;
    data[12] = ONE;
    rowvals[12] = 8;

    0
}

/*
 * Jacobian matrix setup for MGRID >= 4. Translation of `jacHeat`.
 *
 * The C builds the compressed-sparse-column arrays by hand, block by
 * block, with the index arithmetic written out. This follows it
 * statement for statement; the loop bounds and offsets are the C's.
 */

fn jacHeat(
    _tt: sunrealtype,
    cj: sunrealtype,
    _yy: &N_Vector,
    _yp: &N_Vector,
    _resvec: &N_Vector,
    JJ: &SUNMatrix,
    _user_data: &mut Option<Box<dyn Any>>,
    _tempv1: &N_Vector,
    _tempv2: &N_Vector,
    _tempv3: &N_Vector,
) -> i32 {
    let dx = ONE / (MGRID as sunrealtype - ONE);
    let beta = 4.0 / (dx * dx) + cj;
    let mg = MGRID as usize;
    let total = TOTAL as usize;
    let mut repeat: usize;

    SUNMatZero(JJ);

    let mut m = SUNSparseMatrix_Content(JJ);
    let m = &mut *m;
    let (data, rowvals, colptrs) = (&mut m.data, &mut m.indexvals, &mut m.indexptrs);

    /*
     *-----------------------------------------------
     * set up number of elements in each column
     *-----------------------------------------------
     */

    /**** first column block ****/
    colptrs[0] = 0;
    colptrs[1] = 1;
    /* count by twos in the middle  */
    for i in 2..mg {
        colptrs[i] = colptrs[i - 1] + 2;
    }
    colptrs[mg] = 2 * MGRID - 2;

    /**** second column block ****/
    colptrs[mg + 1] = 2 * MGRID;
    colptrs[mg + 2] = 2 * MGRID + 3;
    /* count by fours in the middle */
    for i in 0..mg - 4 {
        colptrs[mg + 3 + i] = colptrs[mg + 3 + i - 1] + 4;
    }
    colptrs[2 * mg - 1] = 2 * MGRID + 4 * (MGRID - 2) - 2;
    colptrs[2 * mg] = 2 * MGRID + 4 * (MGRID - 2);

    /**** repeated (MGRID-4 times) middle column blocks ****/
    repeat = 0;
    for _i in 0..mg - 4 {
        colptrs[2 * mg + 1 + repeat] = colptrs[2 * mg + 1 + repeat - 1] + 2;
        colptrs[2 * mg + 1 + repeat + 1] = colptrs[2 * mg + 1 + repeat] + 4;

        /* count by fives in the middle */
        for j in 0..mg - 4 {
            colptrs[2 * mg + 1 + repeat + 2 + j] = colptrs[2 * mg + 1 + repeat + 1 + j] + 5;
        }

        colptrs[2 * mg + 1 + repeat + (mg - 4) + 2] =
            colptrs[2 * mg + 1 + repeat + (mg - 4) + 1] + 4;
        colptrs[2 * mg + 1 + repeat + (mg - 4) + 3] =
            colptrs[2 * mg + 1 + repeat + (mg - 4) + 2] + 2;

        repeat += mg; /* shift that accounts for accumulated number of columns */
    }

    /**** last-1 column block ****/
    colptrs[mg * mg - 2 * mg + 1] = TOTAL - 2 * MGRID - 4 * (MGRID - 2) + 2;
    colptrs[mg * mg - 2 * mg + 2] = TOTAL - 2 * MGRID - 4 * (MGRID - 2) + 5;
    /* count by fours in the middle */
    for i in 0..mg - 4 {
        colptrs[mg * mg - 2 * mg + 3 + i] = colptrs[mg * mg - 2 * mg + 3 + i - 1] + 4;
    }
    colptrs[mg * mg - mg - 1] = TOTAL - 2 * MGRID;
    colptrs[mg * mg - mg] = TOTAL - 2 * MGRID + 2;

    /**** last column block ****/
    colptrs[mg * mg - mg + 1] = TOTAL - MGRID - (MGRID - 2) + 1;
    /* count by twos in the middle */
    for i in 0..mg - 2 {
        colptrs[mg * mg - mg + 2 + i] = colptrs[mg * mg - mg + 2 + i - 1] + 2;
    }
    colptrs[mg * mg - 1] = TOTAL - 1;
    colptrs[mg * mg] = TOTAL;

    /*
     *-----------------------------------------------
     * set up data stored
     *-----------------------------------------------
     */

    /**** first column block ****/
    data[0] = ONE;
    /* alternating pattern in data, separate loop for each pattern  */
    let mut i = 1;
    while i < mg + (mg - 2) {
        data[i] = ONE;
        i += 2;
    }
    let mut i = 2;
    while i < mg + (mg - 2) - 1 {
        data[i] = -ONE / (dx * dx);
        i += 2;
    }

    /**** second column block ****/
    data[mg + mg - 2] = ONE;
    data[mg + mg - 1] = -ONE / (dx * dx);
    data[mg + mg] = beta;
    data[mg + mg + 1] = -ONE / (dx * dx);
    data[mg + mg + 2] = -ONE / (dx * dx);
    /* middle data elements */
    for i in 0..mg - 4 {
        data[mg + mg + 3 + 4 * i] = -ONE / (dx * dx);
    }
    for i in 0..mg - 4 {
        data[mg + mg + 4 + 4 * i] = beta;
    }
    for i in 0..mg - 4 {
        data[mg + mg + 5 + 4 * i] = -ONE / (dx * dx);
    }
    for i in 0..mg - 4 {
        data[mg + mg + 6 + 4 * i] = -ONE / (dx * dx);
    }
    data[2 * mg + 4 * (mg - 2) - 5] = -ONE / (dx * dx);
    data[2 * mg + 4 * (mg - 2) - 4] = beta;
    data[2 * mg + 4 * (mg - 2) - 3] = -ONE / (dx * dx);
    data[2 * mg + 4 * (mg - 2) - 2] = -ONE / (dx * dx);
    data[2 * mg + 4 * (mg - 2) - 1] = ONE;

    /**** repeated (MGRID-4 times) middle column blocks ****/
    repeat = 0;
    for _i in 0..mg - 4 {
        let b = 2 * mg + 4 * (mg - 2) + repeat;
        data[b] = ONE;
        data[b + 1] = -ONE / (dx * dx);

        data[b + 2] = -ONE / (dx * dx);
        data[b + 3] = beta;
        data[b + 4] = -ONE / (dx * dx);
        data[b + 5] = -ONE / (dx * dx);

        /* 5 in 5*j chosen since there are 5 elements in each column */
        /* this column loops MGRID-4 times within the outer loop */
        for j in 0..mg - 4 {
            data[b + 6 + 5 * j] = -ONE / (dx * dx);
            data[b + 7 + 5 * j] = -ONE / (dx * dx);
            data[b + 8 + 5 * j] = beta;
            data[b + 9 + 5 * j] = -ONE / (dx * dx);
            data[b + 10 + 5 * j] = -ONE / (dx * dx);
        }

        data[b + (mg - 4) * 5 + 6] = -ONE / (dx * dx);
        data[b + (mg - 4) * 5 + 7] = -ONE / (dx * dx);
        data[b + (mg - 4) * 5 + 8] = beta;
        data[b + (mg - 4) * 5 + 9] = -ONE / (dx * dx);

        data[b + (mg - 4) * 5 + 10] = -ONE / (dx * dx);
        data[b + (mg - 4) * 5 + 11] = ONE;

        /* shift that accounts for accumulated columns and elements */
        repeat += mg + 4 * (mg - 2);
    }

    /**** last-1 column block ****/
    data[total - 6 * (mg - 2) - 4] = ONE;
    data[total - 6 * (mg - 2) - 3] = -ONE / (dx * dx);
    data[total - 6 * (mg - 2) - 2] = -ONE / (dx * dx);
    data[total - 6 * (mg - 2) - 1] = beta;
    data[total - 6 * (mg - 2)] = -ONE / (dx * dx);
    /* middle data elements */
    for i in 0..mg - 4 {
        data[total - 6 * (mg - 2) + 1 + 4 * i] = -ONE / (dx * dx);
    }
    for i in 0..mg - 4 {
        data[total - 6 * (mg - 2) + 2 + 4 * i] = -ONE / (dx * dx);
    }
    for i in 0..mg - 4 {
        data[total - 6 * (mg - 2) + 3 + 4 * i] = beta;
    }
    for i in 0..mg - 4 {
        data[total - 6 * (mg - 2) + 4 + 4 * i] = -ONE / (dx * dx);
    }
    data[total - 2 * (mg - 2) - 7] = -ONE / (dx * dx);
    data[total - 2 * (mg - 2) - 6] = -ONE / (dx * dx);
    data[total - 2 * (mg - 2) - 5] = beta;
    data[total - 2 * (mg - 2) - 4] = -ONE / (dx * dx);
    data[total - 2 * (mg - 2) - 3] = ONE;

    /**** last column block ****/
    data[total - 2 * (mg - 2) - 2] = ONE;
    /* alternating pattern in data, separate loop for each pattern  */
    let mut i = total - 2 * (mg - 2) - 1;
    while i < total - 2 {
        data[i] = -ONE / (dx * dx);
        i += 2;
    }
    let mut i = total - 2 * (mg - 2);
    while i < total - 1 {
        data[i] = ONE;
        i += 2;
    }
    data[total - 1] = ONE;

    /*
     *-----------------------------------------------
     * row values
     *-----------------------------------------------
     */

    /**** first block ****/
    rowvals[0] = 0;
    /* alternating pattern in data, separate loop for each pattern */
    let mut i = 1;
    while i < mg + (mg - 2) {
        rowvals[i] = (i as sunindextype + 1) / 2;
        i += 2;
    }
    let mut i = 2;
    while i < mg + (mg - 2) - 1 {
        rowvals[i] = i as sunindextype / 2 + MGRID; /* i+1 unnecessary here */
        i += 2;
    }

    /**** second column block ****/
    rowvals[mg + mg - 2] = MGRID;
    rowvals[mg + mg - 1] = MGRID + 1;
    rowvals[mg + mg] = MGRID + 1;
    rowvals[mg + mg + 1] = MGRID + 2;
    rowvals[mg + mg + 2] = 2 * MGRID + 1;
    /* middle row values */
    for i in 0..mg - 4 {
        rowvals[mg + mg + 3 + 4 * i] = MGRID + 1 + i as sunindextype;
    }
    for i in 0..mg - 4 {
        rowvals[mg + mg + 4 + 4 * i] = MGRID + 2 + i as sunindextype;
    }
    for i in 0..mg - 4 {
        rowvals[mg + mg + 5 + 4 * i] = MGRID + 3 + i as sunindextype;
    }
    for i in 0..mg - 4 {
        rowvals[mg + mg + 6 + 4 * i] = 2 * MGRID + 2 + i as sunindextype;
    }
    rowvals[2 * mg + 4 * (mg - 2) - 5] = MGRID + (MGRID - 2) - 1;
    /* starting from here, add two diag patterns */
    rowvals[2 * mg + 4 * (mg - 2) - 4] = MGRID + (MGRID - 2);
    rowvals[2 * mg + 4 * (mg - 2) - 3] = 2 * MGRID + (MGRID - 2);
    rowvals[2 * mg + 4 * (mg - 2) - 2] = MGRID + (MGRID - 2);
    rowvals[2 * mg + 4 * (mg - 2) - 1] = MGRID + (MGRID - 2) + 1;

    /**** repeated (MGRID-4 times) middle column blocks ****/
    repeat = 0;
    for i in 0..mg - 4 {
        let b = 2 * mg + 4 * (mg - 2) + repeat;
        let ii = i as sunindextype;
        let base = MGRID + (MGRID - 2) + 2 + MGRID * ii;

        rowvals[b] = base;
        rowvals[b + 1] = base + 1;

        rowvals[b + 2] = base + 1 - MGRID;
        rowvals[b + 3] = base + 1;
        rowvals[b + 4] = base + 2; /* *this */
        rowvals[b + 5] = base + 1 + MGRID;

        /* 5 in 5*j chosen since there are 5 elements in each column */
        /* column repeats MGRID-4 times within the outer loop */
        for j in 0..mg - 4 {
            let jj = j as sunindextype;
            rowvals[b + 6 + 5 * j] = base + 1 - MGRID + 1 + jj;
            rowvals[b + 7 + 5 * j] = base + 1 + jj;
            rowvals[b + 8 + 5 * j] = base + 2 + jj;
            rowvals[b + 9 + 5 * j] = base + 2 + 1 + jj;
            rowvals[b + 10 + 5 * j] = base + 1 + MGRID + 1 + jj;
        }

        rowvals[b + (mg - 4) * 5 + 6] = base - 2;
        rowvals[b + (mg - 4) * 5 + 7] = base - 2 + MGRID - 1;
        rowvals[b + (mg - 4) * 5 + 8] = base - 2 + MGRID; /* *this+MGRID */
        rowvals[b + (mg - 4) * 5 + 9] = base - 2 + 2 * MGRID;

        rowvals[b + (mg - 4) * 5 + 10] = base - 2 + MGRID;
        rowvals[b + (mg - 4) * 5 + 11] = base - 2 + MGRID + 1;

        /* shift that accounts for accumulated columns and elements */
        repeat += mg + 4 * (mg - 2);
    }

    /**** last-1 column block ****/
    rowvals[total - 6 * (mg - 2) - 4] = MGRID * MGRID - 1 - 2 * (MGRID - 1) - 1;
    /* starting with this as base */
    rowvals[total - 6 * (mg - 2) - 3] = MGRID * MGRID - 1 - 2 * (MGRID - 1);
    rowvals[total - 6 * (mg - 2) - 2] = MGRID * MGRID - 1 - 2 * (MGRID - 1) - MGRID;
    rowvals[total - 6 * (mg - 2) - 1] = MGRID * MGRID - 1 - 2 * (MGRID - 1);
    rowvals[total - 6 * (mg - 2)] = MGRID * MGRID - 1 - 2 * (MGRID - 1) + 1;
    /* middle row values */
    for i in 0..mg - 4 {
        rowvals[total - 6 * (mg - 2) + 1 + 4 * i] =
            MGRID * MGRID - 1 - 2 * (MGRID - 1) - MGRID + 1 + i as sunindextype;
    }
    for i in 0..mg - 4 {
        rowvals[total - 6 * (mg - 2) + 2 + 4 * i] =
            MGRID * MGRID - 1 - 2 * (MGRID - 1) + i as sunindextype;
    }
    for i in 0..mg - 4 {
        /* copied above */
        rowvals[total - 6 * (mg - 2) + 3 + 4 * i] =
            MGRID * MGRID - 1 - 2 * (MGRID - 1) + 1 + i as sunindextype;
    }
    for i in 0..mg - 4 {
        rowvals[total - 6 * (mg - 2) + 4 + 4 * i] =
            MGRID * MGRID - 1 - 2 * (MGRID - 1) + 2 + i as sunindextype;
    }
    rowvals[total - 2 * (mg - 2) - 7] = MGRID * MGRID - 2 * MGRID - 2;
    rowvals[total - 2 * (mg - 2) - 6] = MGRID * MGRID - MGRID - 3;
    rowvals[total - 2 * (mg - 2) - 5] = MGRID * MGRID - MGRID - 2;
    rowvals[total - 2 * (mg - 2) - 4] = MGRID * MGRID - MGRID - 2;
    rowvals[total - 2 * (mg - 2) - 3] = MGRID * MGRID - MGRID - 1;

    /* last column block */
    rowvals[total - 2 * (mg - 2) - 2] = MGRID * MGRID - MGRID;
    /* alternating pattern in data, separate loop for each pattern  */
    for i in 0..mg - 2 {
        rowvals[total - 2 * (mg - 2) - 1 + 2 * i] =
            MGRID * MGRID - 2 * MGRID + 1 + i as sunindextype;
    }
    for i in 0..mg - 2 {
        rowvals[total - 2 * (mg - 2) + 2 * i] =
            MGRID * MGRID - MGRID + 1 + i as sunindextype;
    }
    rowvals[total - 1] = MGRID * MGRID - 1;

    0
}

/*
 * SetInitialProfile: routine to initialize u, up, and id vectors.
 */

fn SetInitialProfile(
    data: &mut Option<Box<dyn Any>>,
    uu: &N_Vector,
    up: &N_Vector,
    id: &N_Vector,
    res: &N_Vector,
) -> i32 {
    let (mm, dx) = {
        let d = data
            .as_mut()
            .and_then(|b| b.downcast_mut::<UserData>())
            .expect("user_data is UserData");
        (d.mm, d.dx)
    };
    let mm1 = mm - 1;

    /* Initialize id to 1's. */
    N_VConst(ONE, id);

    /* Initialize uu on all grid points. */
    {
        let mut udata = N_VGetArrayPointer(uu).expect("N_VGetArrayPointer");

        for j in 0..mm {
            let yfact = dx * (j as sunrealtype);
            let offset = mm * j;
            for i in 0..mm {
                let xfact = dx * (i as sunrealtype);
                let loc = (offset + i) as usize;
                udata[loc] = 16.0 * xfact * (ONE - xfact) * yfact * (ONE - yfact);
            }
        }
    }

    /* Initialize up vector to 0. */
    N_VConst(ZERO, up);

    /* heatres sets res to negative of ODE RHS values at interior points. */
    heatres(ZERO, uu, up, res, data);

    /* Copy -res into up to get correct interior initial up values. */
    N_VScale(-ONE, res, up);

    /* Finally, set values of u, up, and id at boundary points. */
    {
        let mut udata = N_VGetArrayPointer(uu).expect("N_VGetArrayPointer");
        let mut updata = N_VGetArrayPointer(up).expect("N_VGetArrayPointer");
        let mut iddata = N_VGetArrayPointer(id).expect("N_VGetArrayPointer");

        for j in 0..mm {
            let offset = mm * j;
            for i in 0..mm {
                let loc = (offset + i) as usize;
                if j == 0 || j == mm1 || i == 0 || i == mm1 {
                    udata[loc] = BVAL;
                    updata[loc] = ZERO;
                    iddata[loc] = ZERO;
                }
            }
        }
    }

    0
}

/*
 * Print first lines of output (problem description)
 */

fn PrintHeader(rtol: sunrealtype, atol: sunrealtype) {
    print!("\nidaHeat2D_klu: Heat equation, serial example problem for IDA\n");
    print!("          Discretized heat equation on 2D unit square.\n");
    print!("          Zero boundary conditions,");
    print!(" polynomial initial conditions.\n");
    print!("          Mesh dimensions: {} x {}", MGRID, MGRID);
    print!("        Total system size: {}\n\n", NEQ);
    print!(
        "Tolerance parameters:  rtol = {}   atol = {}\n",
        fmt_g(rtol, 6),
        fmt_g(atol, 6)
    );
    print!("Constraints set to force all solution components >= 0. \n");
    print!("Linear solver: KLU, sparse direct solver \n");
    print!("       difference quotient Jacobian\n");
    print!(
        "IDACalcIC called with input boundary values = {} \n",
        fmt_g(BVAL, 6)
    );
    /* Print output table heading and initial line of table. */
    print!("\n   Output Summary (umax = max-norm of solution) \n\n");
    print!("  time       umax     k  nst  nni  nje   nre     h       \n");
    print!(" .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  . \n");
}

/*
 * Print Output
 */

fn PrintOutput(mem: &IDAMem, t: sunrealtype, uu: &N_Vector) {
    let mut retval: i32;
    let mut hused: sunrealtype = 0.0;
    let mut nst: i64 = 0;
    let mut nni: i64 = 0;
    let mut nje: i64 = 0;
    let mut nre: i64 = 0;
    let mut kused: i32 = 0;

    let umax = N_VMaxNorm(uu);

    retval = IDAGetLastOrder(mem, &mut kused);
    check_retval(&retval, "IDAGetLastOrder");
    retval = IDAGetNumSteps(mem, &mut nst);
    check_retval(&retval, "IDAGetNumSteps");
    retval = IDAGetNumNonlinSolvIters(mem, &mut nni);
    check_retval(&retval, "IDAGetNumNonlinSolvIters");
    retval = IDAGetNumResEvals(mem, &mut nre);
    check_retval(&retval, "IDAGetNumResEvals");
    retval = IDAGetLastStep(mem, &mut hused);
    check_retval(&retval, "IDAGetLastStep");
    retval = IDAGetNumJacEvals(mem, &mut nje);
    check_retval(&retval, "IDAGetNumJacEvals");

    print!(
        " {} {}  {}  {:>3}  {:>3}  {:>3}  {:>4}  {} \n",
        fmt_fw(t, 5, 2),
        fmt_ew(umax, 13, 5),
        kused,
        nst,
        nni,
        nje,
        nre,
        fmt_ew(hused, 9, 2)
    );
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

/* -----------------------------------------------------------------
 * Verification of the assembled Jacobian.
 *
 * `jacHeat` builds its compressed-sparse-column arrays with hand-written
 * index arithmetic, block by block. A transcription slip there produces a
 * matrix that is *almost* right, and the resulting run would look exactly
 * like the ordinary solver divergence this port documents elsewhere --
 * plausible numbers, no error. So the translation is not trusted on
 * inspection; it is checked against two independent constructions.
 *
 * Run with:  cargo test --release -p ida_rs --example idaHeat2D_klu
 * ----------------------------------------------------------------- */

#[cfg(test)]
mod tests {
    use super::*;

    const N: usize = NEQ as usize;
    const M: usize = MGRID as usize;

    fn is_boundary(i: usize) -> bool {
        let (r, c) = (i / M, i % M);
        r == 0 || r == M - 1 || c == 0 || c == M - 1
    }

    /// The Jacobian this example is supposed to assemble, built directly
    /// from the definition rather than from packed indices:
    /// identity on the boundary rows, and `beta` on the diagonal with
    /// `-coeff` on the four stencil neighbours for interior rows.
    fn reference_dense(cj: sunrealtype) -> Vec<Vec<sunrealtype>> {
        let dx = ONE / (MGRID as sunrealtype - ONE);
        let coeff = ONE / (dx * dx);
        let beta = 4.0 / (dx * dx) + cj;
        let mut j = vec![vec![0.0; N]; N];
        for i in 0..N {
            if is_boundary(i) {
                j[i][i] = ONE;
            } else {
                j[i][i] = beta;
                for nb in [i - 1, i + 1, i - M, i + M] {
                    j[i][nb] = -coeff;
                }
            }
        }
        j
    }

    /// Densify whatever `jacHeat` wrote into the sparse matrix.
    fn assembled_dense(cj: sunrealtype) -> Vec<Vec<sunrealtype>> {
        let mut sunctx: Option<SUNContext> = None;
        let _ = SUNContext_Create(SUN_COMM_NULL, &mut sunctx);
        let ctx = sunctx.clone().unwrap();

        let uu = N_VNew_Serial(NEQ, &ctx).unwrap();
        let up = N_VClone(&uu).unwrap();
        let res = N_VClone(&uu).unwrap();
        let a = SUNSparseMatrix(NEQ, NEQ, NEQ * NEQ, SUN_CSC_MAT, &ctx).unwrap();

        let mut ud: Option<Box<dyn Any>> = Some(Box::new(UserData {
            mm: MGRID,
            dx: ONE / (MGRID as sunrealtype - ONE),
            coeff: ONE / ((ONE / (MGRID as sunrealtype - ONE)) * (ONE / (MGRID as sunrealtype - ONE))),
        }));

        assert_eq!(jacHeat(0.0, cj, &uu, &up, &res, &a, &mut ud, &uu, &uu, &uu), 0);

        let m = SUNSparseMatrix_Content(&a);
        let mut dense = vec![vec![0.0; N]; N];
        for col in 0..N {
            for t in m.indexptrs[col] as usize..m.indexptrs[col + 1] as usize {
                dense[m.indexvals[t] as usize][col] += m.data[t];
            }
        }
        dense
    }

    /// The packed arrays must describe exactly the reference matrix.
    #[test]
    fn jacheat_matches_the_definition() {
        let cj = 3.7;
        let got = assembled_dense(cj);
        let want = reference_dense(cj);
        let mut bad = 0;
        for i in 0..N {
            for j in 0..N {
                if (got[i][j] - want[i][j]).abs() > 1e-9 * want[i][j].abs().max(1.0) {
                    if bad < 8 {
                        eprintln!("J[{i}][{j}]: assembled {} vs reference {}", got[i][j], want[i][j]);
                    }
                    bad += 1;
                }
            }
        }
        assert_eq!(bad, 0, "{bad} entries of the assembled Jacobian are wrong");
    }

    /// `TOTAL` must be exactly the number of structural nonzeros, and the
    /// column pointers must be non-decreasing and end at `TOTAL`.
    #[test]
    fn jacheat_nnz_and_colptrs_are_consistent() {
        let want = reference_dense(1.0);
        let nnz: usize = (0..N).map(|i| (0..N).filter(|&j| want[i][j] != 0.0).count()).sum();
        assert_eq!(nnz, TOTAL as usize, "TOTAL disagrees with the structural nonzero count");
    }

    /// The reference itself is not merely asserted: its interior rows are
    /// the finite-difference Jacobian of `heatres`. (Boundary rows are
    /// deliberately identity in this example rather than the true
    /// derivative, because the KLU variant's residual leaves them zero.)
    #[test]
    fn reference_interior_matches_finite_differences() {
        let mut sunctx: Option<SUNContext> = None;
        let _ = SUNContext_Create(SUN_COMM_NULL, &mut sunctx);
        let ctx = sunctx.clone().unwrap();

        let dx = ONE / (MGRID as sunrealtype - ONE);
        let mut ud: Option<Box<dyn Any>> = Some(Box::new(UserData {
            mm: MGRID,
            dx,
            coeff: ONE / (dx * dx),
        }));

        let uu = N_VNew_Serial(NEQ, &ctx).unwrap();
        let up = N_VNew_Serial(NEQ, &ctx).unwrap();
        let r0 = N_VNew_Serial(NEQ, &ctx).unwrap();
        let r1 = N_VNew_Serial(NEQ, &ctx).unwrap();

        /* an arbitrary, non-symmetric state so no term cancels by luck */
        for i in 0..N {
            N_VGetArrayPointer(&uu).unwrap()[i] = 0.3 + 0.7 * (i as sunrealtype) / (N as sunrealtype);
            N_VGetArrayPointer(&up).unwrap()[i] = -0.2 + 0.5 * ((i % 7) as sunrealtype);
        }

        let cj = 3.7;
        let want = reference_dense(cj);
        let h = 1e-6;
        let mut worst = 0.0f64;

        for j in 0..N {
            /* d res / d u_j */
            let u0 = N_VGetArrayPointer(&uu).unwrap()[j];
            heatres(0.0, &uu, &up, &r0, &mut ud);
            N_VGetArrayPointer(&uu).unwrap()[j] = u0 + h;
            heatres(0.0, &uu, &up, &r1, &mut ud);
            N_VGetArrayPointer(&uu).unwrap()[j] = u0;
            let du: Vec<f64> = (0..N)
                .map(|i| {
                    (N_VGetArrayPointer(&r1).unwrap()[i] - N_VGetArrayPointer(&r0).unwrap()[i]) / h
                })
                .collect();

            /* d res / d up_j */
            let p0 = N_VGetArrayPointer(&up).unwrap()[j];
            N_VGetArrayPointer(&up).unwrap()[j] = p0 + h;
            heatres(0.0, &uu, &up, &r1, &mut ud);
            N_VGetArrayPointer(&up).unwrap()[j] = p0;
            let dp: Vec<f64> = (0..N)
                .map(|i| {
                    (N_VGetArrayPointer(&r1).unwrap()[i] - N_VGetArrayPointer(&r0).unwrap()[i]) / h
                })
                .collect();

            for i in 0..N {
                if is_boundary(i) {
                    continue; /* identity by construction, not a derivative */
                }
                let fd = du[i] + cj * dp[i];
                let err = (fd - want[i][j]).abs() / want[i][j].abs().max(1.0);
                if err > worst {
                    worst = err;
                }
                assert!(err < 1e-5, "row {i} col {j}: fd {fd} vs reference {}", want[i][j]);
            }
        }
        eprintln!("reference vs finite differences: worst relative error {worst:.3e}");
    }

    /// Solve with the real Jacobian and check the residual, to separate a
    /// bad matrix from a bad factorization.
    #[test]
    fn sparse_solve_of_this_jacobian_is_accurate() {
        use ida_rs::sundials_sparse_lu::SparseLU;

        let cj = 3.7;
        let mut sunctx: Option<SUNContext> = None;
        let _ = SUNContext_Create(SUN_COMM_NULL, &mut sunctx);
        let ctx = sunctx.clone().unwrap();
        let uu = N_VNew_Serial(NEQ, &ctx).unwrap();
        let up = N_VClone(&uu).unwrap();
        let res = N_VClone(&uu).unwrap();
        let a = SUNSparseMatrix(NEQ, NEQ, NEQ * NEQ, SUN_CSC_MAT, &ctx).unwrap();
        let dx = ONE / (MGRID as sunrealtype - ONE);
        let mut ud: Option<Box<dyn Any>> =
            Some(Box::new(UserData { mm: MGRID, dx, coeff: ONE / (dx * dx) }));
        jacHeat(0.0, cj, &uu, &up, &res, &a, &mut ud, &uu, &uu, &uu);

        let dense = reference_dense(cj);
        let m = SUNSparseMatrix_Content(&a);
        let lu = SparseLU::factor(N, &m.indexptrs, &m.indexvals, &m.data).expect("factor");
        eprintln!("rcond = {:.3e}", lu.rcond());

        let xtrue: Vec<f64> = (0..N).map(|i| 0.5 + (i as f64) / (N as f64)).collect();
        let mut b = vec![0.0; N];
        for i in 0..N {
            for j in 0..N {
                b[i] += dense[i][j] * xtrue[j];
            }
        }
        let mut x = b.clone();
        lu.solve(&mut x);
        let mut worst = 0.0f64;
        for i in 0..N {
            worst = worst.max((x[i] - xtrue[i]).abs() / xtrue[i].abs().max(1.0));
        }
        eprintln!("solve of the heat Jacobian: worst relative error {worst:.3e}");
        assert!(worst < 1e-10, "solve is inaccurate: {worst:e}");
    }
}
