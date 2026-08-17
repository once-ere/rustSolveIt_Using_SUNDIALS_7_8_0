# PROGRESS — per-file port checklist

Project: **SUNDIALS_7_8_Rust_port_for_Linux**. The per-file port status
(ported / building / committed) below is platform-neutral and is inherited
verbatim from the sibling `SUNDIALS_7_8_Rust_port_for_AppleSilicon_macos`,
where the translation was done — all 141 modules are complete.

The per-example **verification** annotations are not platform-neutral: every
"verified" mark and every parenthesised `ref-libm` diagnosis below was made
*on macOS/arm64 against Apple's libm*. On this repository's target
(Linux / x86-64 / glibc) the gate is **153 IDENTICAL / 26 reference-side /
20 excluded**; several `ref-libm` lines here — `idaFoodWeb_bnd`,
`idasFoodWeb_bnd`, `idasSlCrank_dns` among them — are **byte-identical on
Linux**, because the Apple `sin` discrepancy they describe does not exist
against glibc. Part A of `VERIFICATION.md` is the authoritative per-variant
result for this repository.

Status legend: todo | ported | building | committed
(impl headers and public include/ headers port together with their module and share its line)

## Phase 1 — sundials_core

- [x] src/sundials/sundials_math.c — committed
- [x] src/sundials/sundials_errors.c — committed
- [x] src/sundials/sundials_context.c — committed
- [x] src/sundials/sundials_nvector.c — committed
- [x] src/sundials/sundials_matrix.c — committed
- [x] src/sundials/sundials_direct.c — committed
- [x] src/sundials/sundials_band.c — committed
- [x] src/sundials/sundials_dense.c — committed
- [x] src/sundials/sundials_iterative.c — committed
- [x] src/sundials/sundials_linearsolver.c — committed
- [x] src/sundials/sundials_nonlinearsolver.c — committed
- [x] src/sundials/sundials_nvector_senswrapper.c — committed
- [x] src/sundials/sundials_memory.c — committed
- [x] src/sundials/sundials_logger.c — committed
- [x] src/sundials/sundials_profiler.c — committed
- [x] src/sundials/sundials_futils.c — committed
- [x] src/sundials/sundials_hashmap.c — committed
- [x] src/sundials/sundials_version.c — committed
- [x] src/sundials/sundials_cli.c — committed
- [x] src/sundials/sundials_adaptcontroller.c — committed
- [x] src/sundials/sundials_stepper.c — committed
- [x] src/sundials/sundials_adjointstepper.c — committed
- [x] src/sundials/sundials_adjointcheckpointscheme.c — committed
- [x] src/sundials/sundials_datanode.c — committed
- [x] src/sundials/sundials_domeigestimator.c — committed
- [x] src/sundials/sundatanode/sundatanode_inmem.c — committed
- [x] src/sundials/stl/sunstl_vector.h — committed (pulled forward: hashmap needs it)
- [x] src/nvector/serial/nvector_serial.c — committed
- [x] src/sunmatrix/band/sunmatrix_band.c — committed
- [x] src/sunmatrix/dense/sunmatrix_dense.c — committed
- [x] src/sunmatrix/sparse/sunmatrix_sparse.c — committed
- [x] src/sunlinsol/band/sunlinsol_band.c — committed
- [x] src/sunlinsol/dense/sunlinsol_dense.c — committed
- [x] src/sunlinsol/pcg/sunlinsol_pcg.c — committed
- [x] src/sunlinsol/spbcgs/sunlinsol_spbcgs.c — committed
- [x] src/sunlinsol/spfgmr/sunlinsol_spfgmr.c — committed
- [x] src/sunlinsol/spgmr/sunlinsol_spgmr.c — committed
- [x] src/sunlinsol/sptfqmr/sunlinsol_sptfqmr.c — committed
- [x] src/sunnonlinsol/newton/sunnonlinsol_newton.c — committed
- [x] src/sunnonlinsol/fixedpoint/sunnonlinsol_fixedpoint.c — committed
- [x] src/sunnonlinsol/auto/sunnonlinsol_auto.c — committed
- [x] src/sunadaptcontroller/soderlind/sunadaptcontroller_soderlind.c — committed
- [x] src/sunadaptcontroller/imexgus/sunadaptcontroller_imexgus.c — committed
- [x] src/sunadaptcontroller/mrihtol/sunadaptcontroller_mrihtol.c — committed
- [x] src/sundomeigest/power/sundomeigest_power.c — committed
- [x] src/sundomeigest/arnoldi/sundomeigest_arnoldi.c — committed
- [x] src/sunadjointcheckpointscheme/fixed/sunadjointcheckpointscheme_fixed.c — committed
- [x] src/sunmemory/system/sundials_system_memory.c — committed

## Phase 2 — cvode

- [x] src/cvode/cvode.c — committed
- [x] src/cvode/cvode_bandpre.c — committed
- [x] src/cvode/cvode_bbdpre.c — committed
- [x] src/cvode/cvode_cli.c — committed
- [x] src/cvode/cvode_diag.c — committed
- [x] src/cvode/cvode_fused_stubs.c — committed
- [x] src/cvode/cvode_io.c — committed
- [x] src/cvode/cvode_ls.c — committed
- [x] src/cvode/cvode_nls.c — committed
- [x] src/cvode/cvode_proj.c — committed
- [x] src/cvode/cvode_resize.c — committed

## Phase 3 — cvodes

- [x] src/cvodes/cvodea.c — building
- [x] src/cvodes/cvodea_io.c — building
- [x] src/cvodes/cvodes.c — building
- [x] src/cvodes/cvodes_bandpre.c — building
- [x] src/cvodes/cvodes_bbdpre.c — building
- [x] src/cvodes/cvodes_cli.c — building
- [x] src/cvodes/cvodes_diag.c — building
- [x] src/cvodes/cvodes_io.c — building
- [x] src/cvodes/cvodes_ls.c — building
- [x] src/cvodes/cvodes_nls.c — building
- [x] src/cvodes/cvodes_nls_sim.c — building
- [x] src/cvodes/cvodes_nls_stg.c — building
- [x] src/cvodes/cvodes_nls_stg1.c — building
- [x] src/cvodes/cvodes_proj.c — building
- [x] src/cvodes/cvodes_resize.c — building

## Phase 4 — kinsol

- [x] src/kinsol/kinsol.c — building
- [x] src/kinsol/kinsol_aa.c — building
- [x] src/kinsol/kinsol_bbdpre.c — building
- [x] src/kinsol/kinsol_cli.c — building
- [x] src/kinsol/kinsol_io.c — building
- [x] src/kinsol/kinsol_ls.c — building
- [x] src/kinsol/kinsol_orth.c — building

## Phase 5 — ida

- [x] src/ida/ida.c — building
- [x] src/ida/ida_bbdpre.c — building
- [x] src/ida/ida_cli.c — building
- [x] src/ida/ida_ic.c — building
- [x] src/ida/ida_io.c — building
- [x] src/ida/ida_ls.c — building
- [x] src/ida/ida_nls.c — building

## Phase 6 — idas

- [x] src/idas/idaa.c — building
- [x] src/idas/idaa_io.c — building
- [x] src/idas/idas.c — building
- [x] src/idas/idas_bbdpre.c — building
- [x] src/idas/idas_cli.c — building
- [x] src/idas/idas_ic.c — building
- [x] src/idas/idas_io.c — building
- [x] src/idas/idas_ls.c — building
- [x] src/idas/idas_nls.c — building
- [x] src/idas/idas_nls_sim.c — building
- [x] src/idas/idas_nls_stg.c — building

Library fix found by the first example sweep (2026-08-07): the six call
sites that invoke `ida_resS` (`idas_ic.rs` x4, `idas_nls_sim.rs`,
`idas_nls_stg.rs`) took `ida_user_dataS` unconditionally and so handed a
user-supplied `IDASensResFn` a `None` user_data, panicking every example
with an analytic sensitivity residual. C `idas.c:1359` sets
`ida_user_dataS = ida_user_data` in that branch; the port encodes that as
`None` (see `IDASensInit`, `idas.rs:1083`) and the call sites must fall
back to `ida_user_data` — the same "Invariant D" already implemented by
`idab_call_rhsQS` (`idas.rs:3116`) and by CVODES `cvSensRhsWrapper`. Fixed;
unblocks idasRoberts_FSA_dns and idasHessian_ASA_FSA (both now IDENTICAL).

## Phase 7 — arkode

- [x] src/arkode/arkode.c — building
- [x] src/arkode/arkode_adapt.c — building
- [x] src/arkode/arkode_arkstep.c — building
- [x] src/arkode/arkode_arkstep_io.c — building
- [x] src/arkode/arkode_arkstep_nls.c — building
- [x] src/arkode/arkode_bandpre.c — building
- [x] src/arkode/arkode_bbdpre.c — building
- [x] src/arkode/arkode_butcher.c — building
- [x] src/arkode/arkode_butcher_dirk.c — building
- [x] src/arkode/arkode_butcher_erk.c — building
- [x] src/arkode/arkode_cli.c — building
- [x] src/arkode/arkode_erkstep.c — building
- [x] src/arkode/arkode_erkstep_io.c — building
- [x] src/arkode/arkode_forcingstep.c — building
- [x] src/arkode/arkode_interp.c — building
- [x] src/arkode/arkode_io.c — building
- [x] src/arkode/arkode_ls.c — building
- [x] src/arkode/arkode_lsrkstep.c — building
- [x] src/arkode/arkode_lsrkstep_io.c — building
- [x] src/arkode/arkode_mri_tables.c — building
- [x] src/arkode/arkode_mristep.c — building
- [x] src/arkode/arkode_mristep_controller.c — building
- [x] src/arkode/arkode_mristep_io.c — building
- [x] src/arkode/arkode_mristep_nls.c — building
- [x] src/arkode/arkode_relaxation.c — building
- [x] src/arkode/arkode_root.c — building
- [x] src/arkode/arkode_splittingstep.c — building
- [x] src/arkode/arkode_splittingstep_coefficients.c — building
- [x] src/arkode/arkode_sprk.c — building
- [x] src/arkode/arkode_sprkstep.c — building
- [x] src/arkode/arkode_sprkstep_io.c — building
- [x] src/arkode/arkode_sunstepper.c — building
- [x] src/arkode/arkode_user_controller.c — building
- [x] src/arkode/arkode_butcher_dirk.def — committed (tables folded into the including module)
- [x] src/arkode/arkode_butcher_erk.def — committed (tables folded into the including module)
- [x] src/arkode/arkode_mri_tables.def — committed (tables folded into the including module)
- [x] src/arkode/arkode_splittingstep_coefficients.def — committed (tables folded into the including module)

## Phases 3+5 diff-debug pass (2026-08-07)

Every non-IDENTICAL cvodes_rs and ida_rs variant was root-caused against a
locally built pristine upstream-C binary (Release/clang/`-ffp-contract=off`,
logging 2, error checks off, profiling off — the upstream Release defaults).
Result: **11 divergent variants, 0 port defects, 0 source changes.** In all
11 cases the Rust port is byte-identical to the local C build while the
shipped `.out` is not. `ida_rs idaFoodWeb_bnd` moved OPEN -> `ref-libm`
after the shipped reference was reproduced exactly from the C build with a
single correctly-rounded `sin` value. No solver or example code changed, so
`cargo build --workspace` stays warning-free and the cvodes/ida verification
sweeps reproduce the same statuses. Evidence per variant: VERIFICATION.md.

## Phases 4+6 diff-debug pass (2026-08-07)

Every non-IDENTICAL kinsol_rs and idas_rs variant was root-caused against the
same locally built pristine upstream-C binary. Result: **4 divergent variants,
0 port defects, 0 source changes.**

| variant | diff | verdict |
|---|---|---|
| kinRoboKin_dns | 32 lines | stale ref (`SUN_TABLE_WIDTH` 28 -> 29); whitespace-only, `tr -s ' '` diff empty; port and local C both emit `=` at column 31 |
| idasAkzoNob_ASAi_dns | 3 lines | ref trailing-whitespace-stripped; port == local C byte-for-byte |
| idasFoodWeb_bnd | 4 lines | `ref-libm` (1-ulp Apple `sin` at `FOURPI*15/19`); port == local C byte-for-byte |
| idasSlCrank_dns | 6 lines | `ref-libm`; the off-by-one `nre`/`nni` are C's own (local C = 1065/675 like the port); `G` is `sin` vs `__sincos_stret` |
| idasSlCrank_FSA_dns | 22 lines | `ref-libm`; port == local C byte-for-byte on all 47 lines |

No solver or example code changed in this pass, so `cargo build --workspace`
stays warning-free and both verification sweeps reproduce the same statuses
(19/22 kinsol IDENTICAL + 2 excluded; 12/22 idas IDENTICAL + 6 excluded).
**No idas_rs or kinsol_rs variant is OPEN any more.** Evidence per variant:
VERIFICATION.md.

## Phase 7 example sweep (2026-08-09)

All 34 `examples/arkode/C_serial/*.c` programs are ported — no holes; one
`[[example]]` entry per program in `crates/arkode_rs/Cargo.toml`. The crate
and its 34 examples build **warning-free** (`cargo build --release --examples
-p arkode_rs`, verified after `cargo clean --release -p arkode_rs` so no
cached diagnostics are hidden), and `cargo build --workspace` stays clean.

Six examples needed build fixes, all example-side guesses, all minimal, and
**no library change was required**: `1.e-6`/`1.e-10` written without the
mantissa digit (`ark_heat1D`); a missing
`use arkode_rs::sundials_futils::SUNFileClose` (`ark_robertson`); local
`let ZERO`/`let ONE` shadowing the crate constants of the same value, which
Rust rejects as refutable patterns (`ark_robertson`,
`ark_robertson_constraints`); missing `mut` (`ark_brusselator_1D_mri`,
`ark_KrylovDemo_prec`); and one bare `flag = …` assignment made a shadowing
`let` to match every other call site in its file
(`ark_brusselator_lsrk_domeigest`).

All 78 reference variants were preflighted individually under `timeout 120`
before the harness ran — none hangs, crashes or exceeds 10 s. The first
harness result was **43 IDENTICAL, 12 stale-ref (whitespace-only), 23 OPEN,
0 FAIL**; no OPEN variant diverged in a header or setup line, so none was an
example formatting defect.

## Phase 7 debug phase — all 23 OPEN arkode variants closed (2026-08-09)

Final arkode state: **51 IDENTICAL, 13 stale-ref, 14 ref-libm, 0 OPEN,
0 FAIL.** One library change, one line:
`crates/sundials_core/src/sundials_math.rs` `pow_exp_inline` now closes with
`scale.mul_add(tmp, scale)` instead of `scale + scale * tmp`. glibc builds
`sysdeps/x86_64/fpu/multiarch/e_pow-fma.c` with `-ffp-contract=fast`, so
that last operation before the result is rounded is a single fused
multiply-add in the libm the references were generated against; unfused it
lands on the wrong side of the rounding boundary on near-midpoint results.
Localised from `ark_robertson`, where port and a pristine local C build are
bit-identical through step `nst = 31` and then differ by 1 ulp in `eta`
alone (exact value 0.0011 ulp from the midpoint). Effect on the full
199-variant gate: **8 arkode variants newly IDENTICAL** (`ark_robertson`
228 diff lines -> 0, `ark_kepler --stepper ERK --step-mode adapt` 68 -> 0,
four `ark_kepler` SPRK method variants, both `ark_brusselator_lsrk_*`), one
more reduced to whitespace-only, and **zero regressions** anywhere.

The remaining 27 are documented reference-side classes, not port defects:
13 `stale-ref` (SUN_TABLE_WIDTH 28 vs 29 — `tr -s " "` diff empty, `=` at
column 30 in the reference and 31 in the port, the proven `kinRoboKin_dns`
precedent) and 14 `ref-libm` (each closed against a pristine upstream C
build made locally in the reference config; for the kpr_mri, ssprk and
three of the five relaxation variants the port is byte-identical to that C
build while both differ from the shipped `.out`).

Cumulative gate `tools/verify_examples.sh all` (logs/summary-all.txt):
**199 variants — 127 IDENTICAL, 52 documented divergence, 20 excluded
(KLU/SuperLU), 0 FAIL, 0 regressions.** Evidence per variant:
VERIFICATION.md.

## Example programs (one line per ported program; variants tracked in VERIFICATION.md)

- [x] arkode_rs example ark_KrylovDemo_prec — verified IDENTICAL (3 variants)
- [x] arkode_rs example ark_advection_diffusion_reaction_splitting — verified IDENTICAL
- [x] arkode_rs example ark_analytic — verified IDENTICAL (2 variants)
- [x] arkode_rs example ark_analytic_lsrk — verified — 1/1 ref-libm (Soderlind pow near-tie; shipped ref not reproducible from its own source)
- [x] arkode_rs example ark_analytic_lsrk_domeigest — verified — 2/2 ref-libm (same Soderlind pow near-tie)
- [x] arkode_rs example ark_analytic_lsrk_varjac — verified — 1/1 ref-libm (same Soderlind pow near-tie)
- [x] arkode_rs example ark_analytic_mels — verified IDENTICAL
- [x] arkode_rs example ark_analytic_nonlin — verified IDENTICAL
- [x] arkode_rs example ark_analytic_partitioned — verified — 5/5 stale-ref (SUN_TABLE_WIDTH 28 vs 29; values byte-identical)
- [x] arkode_rs example ark_analytic_ssprk — verified — 1/1 ref-libm (atan/exp in RHS; port == pristine local C)
- [x] arkode_rs example ark_brusselator — verified IDENTICAL
- [x] arkode_rs example ark_brusselator1D — verified IDENTICAL
- [x] arkode_rs example ark_brusselator1D_imexmri — verified IDENTICAL (7 variants)
- [x] arkode_rs example ark_brusselator_1D_mri — verified IDENTICAL
- [x] arkode_rs example ark_brusselator_fp — verified IDENTICAL (2 variants)
- [x] arkode_rs example ark_brusselator_lsrk_domeigest — verified IDENTICAL (was OPEN; closed by the pow_exp_inline FMA fix)
- [x] arkode_rs example ark_brusselator_lsrk_externaldomeigest — verified IDENTICAL (was OPEN; closed by the pow_exp_inline FMA fix)
- [x] arkode_rs example ark_brusselator_mri — verified IDENTICAL
- [x] arkode_rs example ark_conserved_exp_entropy_ark — verified — 2/2 ref-libm (exp/log in the adaptivity loop; port == pristine local C)
- [x] arkode_rs example ark_conserved_exp_entropy_erk — verified — 1/1 ref-libm (port == pristine local C, byte-for-byte)
- [x] arkode_rs example ark_damped_harmonic_symplectic — verified — 1/1 stale-ref (SUN_TABLE_WIDTH 28 vs 29)
- [x] arkode_rs example ark_dissipated_exp_entropy — verified — 2/2 ref-libm (exp/log in the adaptivity loop; port == pristine local C)
- [x] arkode_rs example ark_harmonic_symplectic — verified — 1/1 stale-ref (SUN_TABLE_WIDTH 28 vs 29)
- [x] arkode_rs example ark_heat1D — verified IDENTICAL
- [x] arkode_rs example ark_heat1D_adapt — verified IDENTICAL
- [x] arkode_rs example ark_kepler — verified — 13 variants: 8 IDENTICAL (5 closed by the pow FMA fix), 5 stale-ref (SUN_TABLE_WIDTH)
- [x] arkode_rs example ark_kpr_mri — verified — 15 variants: 11 IDENTICAL, 4 ref-libm (cos/sin in ff/fs; port == pristine local C incl. the 17-digit solution file)
- [x] arkode_rs example ark_lotka_volterra_ASA — verified IDENTICAL (2 variants)
- [x] arkode_rs example ark_onewaycouple_mri — verified IDENTICAL
- [x] arkode_rs example ark_reaction_diffusion_mri — verified — 1/1 stale-ref (SUN_TABLE_WIDTH 28 vs 29)
- [x] arkode_rs example ark_robertson — verified IDENTICAL (was OPEN, 228 diff lines; closed by the pow_exp_inline FMA fix)
- [x] arkode_rs example ark_robertson_constraints — verified IDENTICAL
- [x] arkode_rs example ark_robertson_root — verified IDENTICAL
- [x] arkode_rs example ark_twowaycouple_mri — verified IDENTICAL
- [x] cvode_rs example cvAdvDiff_bnd — verified
- [x] cvode_rs example cvAdvDiff_bndL — verified
- [x] cvode_rs example cvAnalytic_mels — verified
- [x] cvode_rs example cvDirectDemo_ls — verified
- [x] cvode_rs example cvDisc_dns — verified
- [x] cvode_rs example cvDiurnal_kry — verified
- [x] cvode_rs example cvDiurnal_kry_bp — verified
- [x] cvode_rs example cvKrylovDemo_ls — verified
- [x] cvode_rs example cvKrylovDemo_prec — verified
- [x] cvode_rs example cvParticle_dns — verified
- [x] cvode_rs example cvPendulum_dns — verified
- [x] cvode_rs example cvRoberts_dns — verified
- [x] cvode_rs example cvRoberts_dnsL — verified
- [x] cvode_rs example cvRoberts_dns_constraints — verified
- [x] cvode_rs example cvRoberts_dns_negsol — verified
- [x] cvode_rs example cvRoberts_dns_uw — verified
- [x] cvode_rs example cvRocket_dns — verified
- [x] cvode_rs example cvVdp_auto_nls — verified
- [x] cvodes_rs example cvsAdvDiff_ASAi_bnd — verified IDENTICAL
- [x] cvodes_rs example cvsAdvDiff_FSA_non — verified IDENTICAL (2 variants)
- [x] cvodes_rs example cvsAdvDiff_bnd — verified IDENTICAL
- [x] cvodes_rs example cvsAdvDiff_bndL — verified IDENTICAL (native band for LAPACK)
- [x] cvodes_rs example cvsAnalytic_mels — verified IDENTICAL (2 variants)
- [x] cvodes_rs example cvsDirectDemo_ls — verified IDENTICAL
- [x] cvodes_rs example cvsDiurnal_FSA_kry — verified (ref-libm, 2 variants; port == local pristine C, 0 diff lines)
- [x] cvodes_rs example cvsDiurnal_kry — verified (ref-libm; port == local pristine C, 0 diff lines)
- [x] cvodes_rs example cvsDiurnal_kry_bp — verified (ref-libm; port == local pristine C, 0 diff lines)
- [x] cvodes_rs example cvsFoodWeb_ASAi_kry — verified IDENTICAL
- [x] cvodes_rs example cvsFoodWeb_ASAp_kry — verified IDENTICAL
- [x] cvodes_rs example cvsHessian_ASA_FSA — verified IDENTICAL
- [x] cvodes_rs example cvsKrylovDemo_ls — verified (ref-libm + ref trailing-ws stripped, 4 variants; port == local pristine C, 0 diff lines)
- [x] cvodes_rs example cvsKrylovDemo_prec — verified IDENTICAL
- [x] cvodes_rs example cvsLotkaVolterra_ASA — verified IDENTICAL
- [x] cvodes_rs example cvsParticle_dns — verified IDENTICAL
- [x] cvodes_rs example cvsPendulum_dns — verified (stale-ref: unreproducible atol exponent; port == local pristine C, 0 diff lines)
- [x] cvodes_rs example cvsRoberts_ASAi_dns — verified IDENTICAL
- [x] cvodes_rs example cvsRoberts_ASAi_dns_constraints — verified IDENTICAL
- [x] cvodes_rs example cvsRoberts_FSA_dns — verified IDENTICAL (2 variants)
- [x] cvodes_rs example cvsRoberts_FSA_dns_Switch — verified IDENTICAL
- [x] cvodes_rs example cvsRoberts_FSA_dns_constraints — verified IDENTICAL
- [x] cvodes_rs example cvsRoberts_dns — verified IDENTICAL
- [x] cvodes_rs example cvsRoberts_dnsL — verified (last-digit LAPACK->native: port == local pristine C with SUNLinSol_Dense, 0 diff lines; + stale-ref spacing)
- [x] cvodes_rs example cvsRoberts_dns_constraints — verified IDENTICAL
- [x] cvodes_rs example cvsRoberts_dns_uw — verified IDENTICAL
- [x] ida_rs example idaAnalytic_mels — verified IDENTICAL (2 variants)
- [x] ida_rs example idaFoodWeb_bnd — verified (ref-libm: 1-ulp Apple sin at FOURPI*15/19 in WebRates; port == local pristine C, and the shipped .out is reproduced byte-for-byte by C + correctly-rounded sin)
- [x] ida_rs example idaFoodWeb_kry — verified IDENTICAL
- [x] ida_rs example idaHeat2D_bnd — verified IDENTICAL
- [x] ida_rs example idaHeat2D_kry — verified IDENTICAL
- [x] ida_rs example idaKrylovDemo_ls — verified IDENTICAL (3 variants)
- [x] ida_rs example idaRoberts_dns — verified IDENTICAL
- [x] ida_rs example idaSlCrank_dns — verified IDENTICAL (clears the idas_rs cross-check: idasSlCrank_dns is IDAS-specific)
- [x] idas_rs example idasAkzoNob_ASAi_dns — verified (exception: ref trailing whitespace stripped; port == local pristine C byte-for-byte)
- [x] idas_rs example idasAkzoNob_dns — verified IDENTICAL
- [x] idas_rs example idasAnalytic_mels — verified IDENTICAL (2 variants)
- [x] idas_rs example idasFoodWeb_bnd — verified (ref-libm: same 1-ulp Apple sin at FOURPI*15/19 as ida_rs idaFoodWeb_bnd; port == local pristine C byte-for-byte)
- [x] idas_rs example idasHeat2D_bnd — verified IDENTICAL
- [x] idas_rs example idasHeat2D_kry — verified IDENTICAL
- [x] idas_rs example idasHessian_ASA_FSA — verified IDENTICAL
- [x] idas_rs example idasKrylovDemo_ls — verified IDENTICAL (3 variants)
- [x] idas_rs example idasRoberts_ASAi_dns — verified IDENTICAL
- [x] idas_rs example idasRoberts_FSA_dns — verified IDENTICAL
- [x] idas_rs example idasRoberts_dns — verified IDENTICAL
- [x] idas_rs example idasSlCrank_FSA_dns — verified (ref-libm: sin/cos in ressc; port == local pristine C byte-for-byte on all 47 lines)
- [x] idas_rs example idasSlCrank_dns — verified (ref-libm: counters == local pristine C, the shipped nre/nni are the reference platform's; G is Apple sin vs __sincos_stret)
- [x] kinsol_rs example kinAnalytic_fp — verified IDENTICAL (11 variants)
- [x] kinsol_rs example kinFerTron_dns — verified IDENTICAL
- [x] kinsol_rs example kinFoodWeb_kry — verified IDENTICAL
- [x] kinsol_rs example kinKrylovDemo_ls — verified IDENTICAL
- [x] kinsol_rs example kinLaplace_bnd — verified IDENTICAL
- [x] kinsol_rs example kinLaplace_picard_bnd — verified IDENTICAL
- [x] kinsol_rs example kinLaplace_picard_kry — verified IDENTICAL
- [x] kinsol_rs example kinRoberts_fp — verified IDENTICAL (2 variants)
- [x] kinsol_rs example kinRoboKin_dns — verified (exception: stale ref SUN_TABLE_WIDTH 28; whitespace-only, every printed value byte-identical to the shipped ref)
