# CVODES — C vs Rust (`examples/cvodes/serial`)

| # | example | argv | class | diff lines / total | worst rel | worst ulp | diff |
|---:|---|---|---|---:|---:|---:|---|
| 1 | `cvsAdvDiff_ASAi_bnd` | _(none)_ | IDENTICAL | — | — | — | — |
| 2 | `cvsAdvDiff_FSA_non` | `-sensi sim t` | IDENTICAL | — | — | — | — |
| 3 | `cvsAdvDiff_FSA_non` | `-sensi stg t` | IDENTICAL | — | — | — | — |
| 4 | `cvsAdvDiff_bnd` | _(none)_ | IDENTICAL | — | — | — | — |
| 5 | `cvsAdvDiff_bndL` | _(none)_ | IDENTICAL | — | — | — | — |
| 6 | `cvsAnalytic_mels` | _(none)_ | IDENTICAL | — | — | — | — |
| 7 | `cvsAnalytic_mels` | `cvodes.max_order 3` | IDENTICAL | — | — | — | — |
| 8 | `cvsDirectDemo_ls` | _(none)_ | IDENTICAL | — | — | — | — |
| 9 | `cvsDiurnal_FSA_kry` | `-sensi sim t` | NUMERIC | 51 / 142 | 1.259e+00 | 9457732915887603712 | [diff](../diffs/cvodes/serial/cvsDiurnal_FSA_kry__-sensi_sim_t.diff) |
| 10 | `cvsDiurnal_FSA_kry` | `-sensi stg t` | IDENTICAL | — | — | — | — |
| 11 | `cvsDiurnal_kry` | _(none)_ | IDENTICAL | — | — | — | — |
| 12 | `cvsDiurnal_kry_bp` | _(none)_ | IDENTICAL | — | — | — | — |
| 13 | `cvsFoodWeb_ASAi_kry` | _(none)_ | IDENTICAL | — | — | — | — |
| 14 | `cvsFoodWeb_ASAp_kry` | _(none)_ | IDENTICAL | — | — | — | — |
| 15 | `cvsHessian_ASA_FSA` | _(none)_ | IDENTICAL | — | — | — | — |
| 16 | `cvsKrylovDemo_ls` | _(none)_ | IDENTICAL | — | — | — | — |
| 17 | `cvsKrylovDemo_ls` | `0 1` | IDENTICAL | — | — | — | — |
| 18 | `cvsKrylovDemo_ls` | `1` | IDENTICAL | — | — | — | — |
| 19 | `cvsKrylovDemo_ls` | `2` | IDENTICAL | — | — | — | — |
| 20 | `cvsKrylovDemo_prec` | _(none)_ | IDENTICAL | — | — | — | — |
| 21 | `cvsLotkaVolterra_ASA` | _(none)_ | IDENTICAL | — | — | — | — |
| 22 | `cvsParticle_dns` | _(none)_ | IDENTICAL | — | — | — | — |
| 23 | `cvsPendulum_dns` | _(none)_ | IDENTICAL | — | — | — | — |
| 24 | `cvsRoberts_ASAi_dns` | _(none)_ | IDENTICAL | — | — | — | — |
| 25 | `cvsRoberts_ASAi_dns_constraints` | _(none)_ | IDENTICAL | — | — | — | — |
| 26 | `cvsRoberts_ASAi_klu` | _(none)_ | NUMERIC | 7 / 60 | 6.015e-02 | 387028092977152 | [diff](../diffs/cvodes/serial/cvsRoberts_ASAi_klu.diff) |
| 27 | `cvsRoberts_ASAi_sps` | _(none)_ | NOT_PORTED | — | — | — | — |
| 28 | `cvsRoberts_FSA_dns` | `-sensi sim t` | IDENTICAL | — | — | — | — |
| 29 | `cvsRoberts_FSA_dns` | `-sensi stg1 t` | IDENTICAL | — | — | — | — |
| 30 | `cvsRoberts_FSA_dns_Switch` | _(none)_ | IDENTICAL | — | — | — | — |
| 31 | `cvsRoberts_FSA_dns_constraints` | `-sensi stg1 t` | IDENTICAL | — | — | — | — |
| 32 | `cvsRoberts_FSA_klu` | `-sensi stg1 t` | NUMERIC | 1 / 87 | 5.464e-05 | 371382011785 | [diff](../diffs/cvodes/serial/cvsRoberts_FSA_klu__-sensi_stg1_t.diff) |
| 33 | `cvsRoberts_FSA_sps` | `-sensi stg1 t` | NOT_PORTED | — | — | — | — |
| 34 | `cvsRoberts_dns` | _(none)_ | IDENTICAL | — | — | — | — |
| 35 | `cvsRoberts_dnsL` | _(none)_ | IDENTICAL | — | — | — | — |
| 36 | `cvsRoberts_dns_constraints` | _(none)_ | IDENTICAL | — | — | — | — |
| 37 | `cvsRoberts_dns_uw` | _(none)_ | IDENTICAL | — | — | — | — |
| 38 | `cvsRoberts_klu` | _(none)_ | NUMERIC | 8 / 24 | 3.333e-01 | 2251799813685248 | [diff](../diffs/cvodes/serial/cvsRoberts_klu.diff) |
| 39 | `cvsRoberts_sps` | _(none)_ | NOT_PORTED | — | — | — | — |

