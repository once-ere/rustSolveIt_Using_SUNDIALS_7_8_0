# IDAS — C vs Rust (`examples/idas/serial`)

| # | example | argv | class | diff lines / total | worst rel | worst ulp | diff |
|---:|---|---|---|---:|---:|---:|---|
| 1 | `idasAkzoNob_ASAi_dns` | _(none)_ | IDENTICAL | — | — | — | — |
| 2 | `idasAkzoNob_dns` | _(none)_ | IDENTICAL | — | — | — | — |
| 3 | `idasAnalytic_mels` | _(none)_ | IDENTICAL | — | — | — | — |
| 4 | `idasAnalytic_mels` | `idas.init_step 1e-5` | IDENTICAL | — | — | — | — |
| 5 | `idasFoodWeb_bnd` | _(none)_ | IDENTICAL | — | — | — | — |
| 6 | `idasHeat2D_bnd` | _(none)_ | IDENTICAL | — | — | — | — |
| 7 | `idasHeat2D_kry` | _(none)_ | IDENTICAL | — | — | — | — |
| 8 | `idasHessian_ASA_FSA` | _(none)_ | IDENTICAL | — | — | — | — |
| 9 | `idasKrylovDemo_ls` | _(none)_ | IDENTICAL | — | — | — | — |
| 10 | `idasKrylovDemo_ls` | `1` | IDENTICAL | — | — | — | — |
| 11 | `idasKrylovDemo_ls` | `2` | IDENTICAL | — | — | — | — |
| 12 | `idasRoberts_ASAi_dns` | _(none)_ | IDENTICAL | — | — | — | — |
| 13 | `idasRoberts_ASAi_klu` | _(none)_ | NUMERIC | 3 / 37 | 9.098e-02 | 527765581332480 | [diff](../diffs/idas/serial/idasRoberts_ASAi_klu.diff) |
| 14 | `idasRoberts_ASAi_sps` | _(none)_ | NOT_PORTED | — | — | — | — |
| 15 | `idasRoberts_FSA_dns` | `-sensi stg t` | IDENTICAL | — | — | — | — |
| 16 | `idasRoberts_FSA_klu` | `-sensi stg t` | NUMERIC | 34 / 117 | 5.720e-01 | 5500856673763328 | [diff](../diffs/idas/serial/idasRoberts_FSA_klu__-sensi_stg_t.diff) |
| 17 | `idasRoberts_FSA_sps` | `-sensi stg t` | NOT_PORTED | — | — | — | — |
| 18 | `idasRoberts_dns` | _(none)_ | IDENTICAL | — | — | — | — |
| 19 | `idasRoberts_klu` | _(none)_ | IDENTICAL | — | — | — | — |
| 20 | `idasRoberts_sps` | _(none)_ | NOT_PORTED | — | — | — | — |
| 21 | `idasSlCrank_FSA_dns` | _(none)_ | NUMERIC | 8 / 38 | 1.000e+00 | 4607182418800017408 | [diff](../diffs/idas/serial/idasSlCrank_FSA_dns.diff) |
| 22 | `idasSlCrank_dns` | _(none)_ | NUMERIC | 1 / 47 | 7.420e-13 | 5575 | [diff](../diffs/idas/serial/idasSlCrank_dns.diff) |

