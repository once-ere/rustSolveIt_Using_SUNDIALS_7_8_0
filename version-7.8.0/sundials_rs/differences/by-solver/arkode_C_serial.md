# ARKODE — C vs Rust (`examples/arkode/C_serial`)

| # | example | argv | class | diff lines / total | worst rel | worst ulp | diff |
|---:|---|---|---|---:|---:|---:|---|
| 1 | `ark_KrylovDemo_prec` | _(none)_ | IDENTICAL | — | — | — | — |
| 2 | `ark_KrylovDemo_prec` | `1` | IDENTICAL | — | — | — | — |
| 3 | `ark_KrylovDemo_prec` | `2` | IDENTICAL | — | — | — | — |
| 4 | `ark_advection_diffusion_reaction_splitting` | _(none)_ | IDENTICAL | — | — | — | — |
| 5 | `ark_analytic` | _(none)_ | IDENTICAL | — | — | — | — |
| 6 | `ark_analytic` | `arkode.scalar_tolerances 1e-6 1e-8 arkode.table_names ARKODE_ESDIRK547L2SA_7_4_5 ARKODE_ERK_NONE` | IDENTICAL | — | — | — | — |
| 7 | `ark_analytic_lsrk` | _(none)_ | NUMERIC | 7 / 40 | 2.818e-01 | 1873581158628234 | [diff](../diffs/arkode/C_serial/ark_analytic_lsrk.diff) |
| 8 | `ark_analytic_lsrk_domeigest` | _(none)_ | NUMERIC | 14 / 46 | 2.600e-01 | 2020902451812283 | [diff](../diffs/arkode/C_serial/ark_analytic_lsrk_domeigest.diff) |
| 9 | `ark_analytic_lsrk_domeigest` | `arkid.dom_eig_est_init_preprocess_iters 1 sundomeigestimator.max_iters 1` | NUMERIC | 14 / 46 | 5.995e-01 | 6205658242664437 | [diff](../diffs/arkode/C_serial/ark_analytic_lsrk_domeigest__arkid.dom_eig_est_init_preprocess_iters_1_sundomeigestimator.max_iters_1.diff) |
| 10 | `ark_analytic_lsrk_varjac` | _(none)_ | NUMERIC | 12 / 44 | 2.036e-01 | 1807600431283255 | [diff](../diffs/arkode/C_serial/ark_analytic_lsrk_varjac.diff) |
| 11 | `ark_analytic_mels` | _(none)_ | IDENTICAL | — | — | — | — |
| 12 | `ark_analytic_nonlin` | _(none)_ | IDENTICAL | — | — | — | — |
| 13 | `ark_analytic_partitioned` | `forcing` | IDENTICAL | — | — | — | — |
| 14 | `ark_analytic_partitioned` | `splitting` | IDENTICAL | — | — | — | — |
| 15 | `ark_analytic_partitioned` | `splitting ARKODE_SPLITTING_BEST_2_2_2` | IDENTICAL | — | — | — | — |
| 16 | `ark_analytic_partitioned` | `splitting ARKODE_SPLITTING_RUTH_3_3_2` | IDENTICAL | — | — | — | — |
| 17 | `ark_analytic_partitioned` | `splitting ARKODE_SPLITTING_YOSHIDA_8_6_2` | IDENTICAL | — | — | — | — |
| 18 | `ark_analytic_ssprk` | _(none)_ | IDENTICAL | — | — | — | — |
| 19 | `ark_brusselator` | _(none)_ | IDENTICAL | — | — | — | — |
| 20 | `ark_brusselator1D` | _(none)_ | IDENTICAL | — | — | — | — |
| 21 | `ark_brusselator1D_imexmri` | `0 0.001` | IDENTICAL | — | — | — | — |
| 22 | `ark_brusselator1D_imexmri` | `2 0.001` | IDENTICAL | — | — | — | — |
| 23 | `ark_brusselator1D_imexmri` | `3 0.001` | IDENTICAL | — | — | — | — |
| 24 | `ark_brusselator1D_imexmri` | `4 0.001` | IDENTICAL | — | — | — | — |
| 25 | `ark_brusselator1D_imexmri` | `5 0.001` | IDENTICAL | — | — | — | — |
| 26 | `ark_brusselator1D_imexmri` | `6 0.001` | IDENTICAL | — | — | — | — |
| 27 | `ark_brusselator1D_imexmri` | `7 0.001` | IDENTICAL | — | — | — | — |
| 28 | `ark_brusselator_1D_mri` | _(none)_ | IDENTICAL | — | — | — | — |
| 29 | `ark_brusselator_fp` | _(none)_ | IDENTICAL | — | — | — | — |
| 30 | `ark_brusselator_fp` | `1` | IDENTICAL | — | — | — | — |
| 31 | `ark_brusselator_lsrk_domeigest` | _(none)_ | IDENTICAL | — | — | — | — |
| 32 | `ark_brusselator_lsrk_externaldomeigest` | _(none)_ | IDENTICAL | — | — | — | — |
| 33 | `ark_brusselator_mri` | _(none)_ | IDENTICAL | — | — | — | — |
| 34 | `ark_conserved_exp_entropy_ark` | `1 0` | IDENTICAL | — | — | — | — |
| 35 | `ark_conserved_exp_entropy_ark` | `1 1` | IDENTICAL | — | — | — | — |
| 36 | `ark_conserved_exp_entropy_erk` | `1` | IDENTICAL | — | — | — | — |
| 37 | `ark_damped_harmonic_symplectic` | _(none)_ | IDENTICAL | — | — | — | — |
| 38 | `ark_dissipated_exp_entropy` | `1 0` | IDENTICAL | — | — | — | — |
| 39 | `ark_dissipated_exp_entropy` | `1 1` | IDENTICAL | — | — | — | — |
| 40 | `ark_harmonic_symplectic` | _(none)_ | IDENTICAL | — | — | — | — |
| 41 | `ark_heat1D` | _(none)_ | IDENTICAL | — | — | — | — |
| 42 | `ark_heat1D_adapt` | _(none)_ | IDENTICAL | — | — | — | — |
| 43 | `ark_kepler` | _(none)_ | IDENTICAL | — | — | — | — |
| 44 | `ark_kepler` | `--stepper ERK --step-mode adapt` | IDENTICAL | — | — | — | — |
| 45 | `ark_kepler` | `--stepper ERK --step-mode fixed --count-orbits` | IDENTICAL | — | — | — | — |
| 46 | `ark_kepler` | `--stepper SPRK --step-mode fixed --count-orbits --use-compensated-sums` | IDENTICAL | — | — | — | — |
| 47 | `ark_kepler` | `--stepper SPRK --step-mode fixed --method ARKODE_SPRK_EULER_1_1 --tf 50 --check-order --nout 1` | IDENTICAL | — | — | — | — |
| 48 | `ark_kepler` | `--stepper SPRK --step-mode fixed --method ARKODE_SPRK_LEAPFROG_2_2 --tf 50 --check-order --nout 1` | IDENTICAL | — | — | — | — |
| 49 | `ark_kepler` | `--stepper SPRK --step-mode fixed --method ARKODE_SPRK_MCLACHLAN_2_2 --tf 50 --check-order --nout 1` | IDENTICAL | — | — | — | — |
| 50 | `ark_kepler` | `--stepper SPRK --step-mode fixed --method ARKODE_SPRK_MCLACHLAN_3_3 --tf 50 --check-order --nout 1` | IDENTICAL | — | — | — | — |
| 51 | `ark_kepler` | `--stepper SPRK --step-mode fixed --method ARKODE_SPRK_MCLACHLAN_4_4 --tf 50 --check-order --nout 1` | IDENTICAL | — | — | — | — |
| 52 | `ark_kepler` | `--stepper SPRK --step-mode fixed --method ARKODE_SPRK_MCLACHLAN_5_6 --tf 50 --check-order --nout 1` | IDENTICAL | — | — | — | — |
| 53 | `ark_kepler` | `--stepper SPRK --step-mode fixed --method ARKODE_SPRK_PSEUDO_LEAPFROG_2_2 --tf 50 --check-order --nout 1` | IDENTICAL | — | — | — | — |
| 54 | `ark_kepler` | `--stepper SPRK --step-mode fixed --method ARKODE_SPRK_RUTH_3_3 --tf 50 --check-order --nout 1` | IDENTICAL | — | — | — | — |
| 55 | `ark_kepler` | `--stepper SPRK --step-mode fixed --method ARKODE_SPRK_YOSHIDA_6_8 --tf 50 --check-order --nout 1` | IDENTICAL | — | — | — | — |
| 56 | `ark_kpr_mri` | `0 1 0.005` | IDENTICAL | — | — | — | — |
| 57 | `ark_kpr_mri` | `1 0 0.01` | IDENTICAL | — | — | — | — |
| 58 | `ark_kpr_mri` | `1 1 0.002` | IDENTICAL | — | — | — | — |
| 59 | `ark_kpr_mri` | `10 4 0.001 -100 100 0.5 1` | NUMERIC | 1 / 73 | 1.266e-03 | 9903520314283 | [diff](../diffs/arkode/C_serial/ark_kpr_mri__10_4_0.001_-100_100_0.5_1.diff) |
| 60 | `ark_kpr_mri` | `11 2 0.001` | IDENTICAL | — | — | — | — |
| 61 | `ark_kpr_mri` | `12 3 0.005` | IDENTICAL | — | — | — | — |
| 62 | `ark_kpr_mri` | `13 4 0.01` | IDENTICAL | — | — | — | — |
| 63 | `ark_kpr_mri` | `2 4 0.002` | IDENTICAL | — | — | — | — |
| 64 | `ark_kpr_mri` | `3 2 0.001` | IDENTICAL | — | — | — | — |
| 65 | `ark_kpr_mri` | `4 3 0.001` | IDENTICAL | — | — | — | — |
| 66 | `ark_kpr_mri` | `5 4 0.001` | IDENTICAL | — | — | — | — |
| 67 | `ark_kpr_mri` | `6 5 0.001` | IDENTICAL | — | — | — | — |
| 68 | `ark_kpr_mri` | `7 2 0.002` | IDENTICAL | — | — | — | — |
| 69 | `ark_kpr_mri` | `8 3 0.001 -100 100 0.5 1` | IDENTICAL | — | — | — | — |
| 70 | `ark_kpr_mri` | `9 3 0.001 -100 100 0.5 1` | IDENTICAL | — | — | — | — |
| 71 | `ark_lotka_volterra_ASA` | `--check-freq 1` | IDENTICAL | — | — | — | — |
| 72 | `ark_lotka_volterra_ASA` | `--check-freq 5` | IDENTICAL | — | — | — | — |
| 73 | `ark_onewaycouple_mri` | _(none)_ | IDENTICAL | — | — | — | — |
| 74 | `ark_reaction_diffusion_mri` | _(none)_ | IDENTICAL | — | — | — | — |
| 75 | `ark_robertson` | _(none)_ | IDENTICAL | — | — | — | — |
| 76 | `ark_robertson_constraints` | _(none)_ | IDENTICAL | — | — | — | — |
| 77 | `ark_robertson_root` | _(none)_ | IDENTICAL | — | — | — | — |
| 78 | `ark_twowaycouple_mri` | _(none)_ | IDENTICAL | — | — | — | — |

